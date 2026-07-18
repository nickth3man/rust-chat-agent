#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    openrouter_chat_rust::runtime::run().await
}
