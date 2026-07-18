use openrouter_chat_rust::config::ProviderConfig;
use openrouter_chat_rust::contracts::ToolNetError;
use openrouter_chat_rust::search::BackendHttp;
use openrouter_chat_rust::search::backends::{duckduckgo, hn};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn cfg(base: String) -> ProviderConfig {
    ProviderConfig {
        enable: true,
        api_key_env: None,
        optional_api_key_env: None,
        concurrency: 1,
        min_interval_ms: 0,
        timeout_secs: 1,
        user_agent: "p2-fixture/1 (+https://example.invalid)".into(),
        base_url: Some(base),
        base_url_env: None,
    }
}

struct Server {
    address: String,
    requests: Arc<Mutex<Vec<String>>>,
    join: thread::JoinHandle<()>,
}

fn read_request(stream: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut buf = [0; 1024];
    loop {
        let n = stream.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
        if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
            let header_end = bytes
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .unwrap()
                + 4;
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let length = headers
                .lines()
                .find_map(|line| {
                    line.strip_prefix("Content-Length:")?
                        .trim()
                        .parse::<usize>()
                        .ok()
                })
                .unwrap_or(0);
            while bytes.len() < header_end + length {
                let n = stream.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                bytes.extend_from_slice(&buf[..n]);
            }
            break;
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn spawn_server(responses: Vec<String>, pause: Option<Duration>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = requests.clone();
    let join = thread::spawn(move || {
        for response in responses {
            let (mut stream, _) = listener.accept().unwrap();
            captured.lock().unwrap().push(read_request(&mut stream));
            if let Some(delay) = pause {
                thread::sleep(delay);
            }
            stream.write_all(response.as_bytes()).unwrap();
        }
    });
    Server {
        address,
        requests,
        join,
    }
}

fn response(status: &str, headers: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{headers}\r\n{body}",
        body.len()
    )
}

async fn http(server: &Server) -> BackendHttp {
    BackendHttp::new("fixture", &cfg(server.address.clone()), None)
        .unwrap()
        .with_limits(2 * 1024 * 1024, Duration::from_secs(2))
        .with_retry_backoff(Duration::from_millis(5))
}

#[tokio::test]
async fn get_encodes_query_and_sets_user_agent() {
    let server = spawn_server(vec![response("200 OK", "", "{}")], None);
    let client = http(&server).await;
    let _: serde_json::Value = client
        .get_json(&client.url("/search"), &[("q", "a b&c")])
        .await
        .unwrap();
    server.join.join().unwrap();
    let request = server.requests.lock().unwrap()[0].clone();
    assert!(request.starts_with("GET /search?q=a+b%26c HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("user-agent: p2-fixture/1")
    );
}

#[tokio::test]
async fn post_form_and_custom_auth_are_transmitted_without_debug_leak() {
    let server = spawn_server(vec![response("200 OK", "", "ok")], None);
    let client = http(&server).await;
    let text = client
        .post_form_text(&client.url("/form"), &[("q", "a b")])
        .await
        .unwrap();
    assert_eq!(text, "ok");
    server.join.join().unwrap();
    let request = server.requests.lock().unwrap()[0].clone();
    assert!(
        request
            .to_ascii_lowercase()
            .contains("content-type: application/x-www-form-urlencoded")
    );
    assert!(request.ends_with("q=a+b"));

    let server = spawn_server(vec![response("200 OK", "", "{}")], None);
    let client = http(&server).await;
    let _: serde_json::Value = client
        .get_json_with_headers(
            &client.url("/auth"),
            &[],
            &[("X-Subscription-Token", "secret-value")],
        )
        .await
        .unwrap();
    assert!(!format!("{client:?}").contains("secret-value"));
    server.join.join().unwrap();
    assert!(
        server.requests.lock().unwrap()[0]
            .to_ascii_lowercase()
            .contains("x-subscription-token: secret-value")
    );
}

#[tokio::test]
async fn retry_after_zero_retries_once_and_succeeds() {
    let server = spawn_server(
        vec![
            response("429 Too Many Requests", "Retry-After: 0\r\n", "busy"),
            response("200 OK", "", "{}"),
        ],
        None,
    );
    let client = http(&server).await;
    let _: serde_json::Value = client.get_json(&client.url("/retry"), &[]).await.unwrap();
    server.join.join().unwrap();
    assert_eq!(server.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn absent_retry_after_uses_backoff_and_terminal_retry_is_exactly_once() {
    let server = spawn_server(
        vec![
            response("503 Service Unavailable", "", "busy"),
            response("503 Service Unavailable", "", "still busy"),
        ],
        None,
    );
    let client = http(&server).await;
    let started = Instant::now();
    assert!(client.get_text(&client.url("/retry"), &[]).await.is_err());
    server.join.join().unwrap();
    assert_eq!(server.requests.lock().unwrap().len(), 2);
    assert!(started.elapsed() >= Duration::from_millis(5));
}

#[tokio::test]
async fn not_found_is_not_retried_and_body_cap_is_enforced() {
    let server = spawn_server(vec![response("404 Not Found", "", "missing")], None);
    let client = http(&server).await;
    assert!(client.get_text(&client.url("/missing"), &[]).await.is_err());
    server.join.join().unwrap();
    assert_eq!(server.requests.lock().unwrap().len(), 1);

    let server = spawn_server(vec![response("200 OK", "", "0123456789")], None);
    let client = http(&server).await.with_limits(4, Duration::from_secs(1));
    assert!(matches!(
        client.get_text(&client.url("/big"), &[]).await,
        Err(ToolNetError::BodyTooLarge { limit: 4 })
    ));
    server.join.join().unwrap();
}

#[tokio::test]
async fn redirect_is_terminal_and_does_not_forward_custom_auth() {
    let target = TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let target_address = format!("http://{}", target.local_addr().unwrap());
    let server = spawn_server(
        vec![response(
            "302 Found",
            &format!("Location: {target_address}/target\r\n"),
            "redirect",
        )],
        None,
    );
    let client = http(&server).await;
    let result = client
        .get_json_with_headers::<serde_json::Value>(
            &client.url("/redirect"),
            &[],
            &[("X-Subscription-Token", "secret-value")],
        )
        .await;

    assert!(matches!(
        result,
        Err(ToolNetError::HttpStatus { status: 302, .. })
    ));
    server.join.join().unwrap();
    assert_eq!(server.requests.lock().unwrap().len(), 1);

    let deadline = Instant::now() + Duration::from_millis(100);
    let mut target_request = None;
    while Instant::now() < deadline {
        match target.accept() {
            Ok((mut stream, _)) => {
                target_request = Some(read_request(&mut stream));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("target listener failed: {error}"),
        }
    }
    assert!(target_request.is_none());
}

#[tokio::test]
async fn timeout_maps_to_timeout() {
    let server = spawn_server(
        vec![response("200 OK", "", "ok")],
        Some(Duration::from_millis(100)),
    );
    let client = http(&server)
        .await
        .with_timeout(Duration::from_millis(10))
        .unwrap();
    assert!(matches!(
        client.get_text(&client.url("/slow"), &[]).await,
        Err(ToolNetError::Timeout)
    ));
    server.join.join().unwrap();
}

#[test]
fn retry_after_parser_and_hn_mapping_are_precise() {
    assert_eq!(
        ToolNetError::parse_retry_after(Some("0")),
        Some(Duration::ZERO)
    );
    let json = r#"{"hits":[{"title":null,"story_title":"Story","url":null,"story_url":"https://article","author":"alice","points":7,"created_at":"today","objectID":"42"}]}"#;
    let hits = hn::parse_json(json, 5).unwrap();
    assert_eq!(hits[0].title, "Story");
    assert_eq!(hits[0].url, "https://article");
    assert_eq!(hits[0].native_rank, Some(1));
    assert_eq!(hits[0].native_score, Some(7.0));
    assert_eq!(hits[0].published.as_deref(), Some("today"));
    assert_eq!(
        hits[0].metadata["discussion_url"],
        "https://news.ycombinator.com/item?id=42"
    );
}

#[test]
fn duckduckgo_lite_maps_sibling_snippet_and_relative_redirect() {
    let html = r#"<table><tr><td><a class="result-link" href="/l/?uddg=https%3A%2F%2Fexample.com%2F">Title</a></td><td class="result-snippet">A sibling snippet</td></tr></table>"#;
    let hits = duckduckgo::parse_results(html, 5);
    assert_eq!(hits[0].url, "https://example.com/");
    assert_eq!(hits[0].snippet, "A sibling snippet");
}
