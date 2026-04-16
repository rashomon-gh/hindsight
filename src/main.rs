//! Hindsight — agentic memory architecture for AI agents.
//!
//! This binary runs an interactive REPL that accepts user messages, stores
//! extracted facts into a structured memory bank, recalls relevant context,
//! and generates preference-conditioned responses.
//!
//! See the module-level docs for each subsystem:
//!
//! - [`config`] — YAML-based configuration
//! - [`models`] — data structures (networks, edges, profile)
//! - [`llm`] — OpenAI-compatible HTTP client
//! - [`storage`] — PostgreSQL + pgvector persistence
//! - [`tempr`] — Retain & Recall pipeline
//! - [`cara`] — Reflect pipeline

mod api;
mod cara;
mod config;
mod files;
mod llm;
mod models;
mod storage;
mod tempr;

use std::io::{self, Write};
use std::sync::Arc;
use std::env;

use anyhow::Result;
use cara::CaraPipeline;
use config::Config;
use llm::LLMClient;
use models::AgentProfile;
use storage::Storage;
use tempr::TemprPipeline;
use api::WebServer;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hindsight=info".into()),
        )
        .init();

    let config = Config::load()?;

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    let web_mode = args.len() > 1 && (args[1] == "--web" || args[1] == "--serve");
    let cli_mode = !web_mode || (args.len() > 2 && args[2] == "--cli");

    println!("Connecting to database at {}...", config.database.url);
    let storage = Arc::new(Storage::connect(&config.database.url).await?);
    storage.init_schema().await?;
    println!("Database schema initialized.");

    let llm = Arc::new(LLMClient::new(&config.llm));
    let embedding_dim = config.llm.embedding_dim;

    // Create a new storage connection for TEMPR (it takes ownership)
    let storage_for_tempr = Storage::connect(&config.database.url).await?;
    let llm_for_tempr = LLMClient::new(&config.llm);
    let tempr = TemprPipeline::new(llm_for_tempr, storage_for_tempr, embedding_dim);

    let profile = AgentProfile {
        name: "Hindsight".into(),
        background: "I am an AI agent with a structured long-term memory system. I can retain, recall, and reflect on information across conversations.".into(),
        skepticism: 3,
        literalism: 2,
        empathy: 4,
        bias_strength: 0.5,
    };

    let cara = CaraPipeline::new(profile, tempr);

    // Start web server if enabled or requested
    if web_mode || config.web.enabled {
        let web_host = config.web.host.clone();
        let web_port = config.web.port;

        let web_config = api::WebConfig {
            host: web_host.clone(),
            port: web_port,
            enabled: true,
        };

        let web_server = WebServer::new(web_config, storage.clone(), llm, embedding_dim);

        if cli_mode {
            // Run both CLI and web server
            println!("🌐 Starting web server at http://{}:{}...", web_host, web_port);
            println!("Hindsight agent ready. Type a message (or 'quit' to exit):");

            let (shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

            // Handle Ctrl+C for both CLI and web server
            let shutdown_tx_clone = shutdown_tx.clone();
            tokio::spawn(async move {
                match tokio::signal::ctrl_c().await {
                    Ok(()) => {
                        println!("\n\nReceived Ctrl+C, shutting down...");
                    }
                    Err(err) => {
                        tracing::error!("Failed to install Ctrl+C handler: {}", err);
                    }
                }
                let _ = shutdown_tx_clone.send(());
            });

            let web_handle = tokio::spawn(async move {
                if let Err(e) = web_server.run().await {
                    tracing::error!("Web server error: {}", e);
                }
            });

            // Run CLI REPL with shutdown handling
            let cli_result = tokio::select! {
                _ = shutdown_rx.recv() => {
                    println!("CLI shutdown signal received");
                    Ok(())
                }
                result = run_cli_repl(cara) => result,
            };

            // Cancel web server when CLI exits or is interrupted
            web_handle.abort();
            cli_result?;

        } else {
            // Run web server only
            web_server.run().await?;
            return Ok(());
        }
    } else {
        println!("Hindsight agent ready. Type a message (or 'quit' to exit):\n");
        run_cli_repl(cara).await?;
    }

    println!("Goodbye!");
    Ok(())
}

/// Run the interactive CLI REPL.
async fn run_cli_repl(cara: CaraPipeline) -> Result<()> {
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

    // Spawn a task to handle Ctrl+C for the CLI
    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                println!("\n\nReceived Ctrl+C, shutting down CLI...");
            }
            Err(err) => {
                tracing::error!("Failed to install Ctrl+C handler: {}", err);
            }
        }
        let _ = shutdown_tx_clone.send(());
    });

    loop {
        print!("> ");
        io::stdout().flush()?;

        // Use select to wait for either input or shutdown signal
        tokio::select! {
            _ = shutdown_rx.recv() => {
                println!("CLI shutdown signal received");
                break;
            }
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                // Small delay to allow shutdown signal to be processed
                let stdin = io::stdin();
                let mut input = String::new();

                match stdin.read_line(&mut input) {
                    Ok(_) => {
                        let input = input.trim();

                        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
                            break;
                        }

                        if !input.is_empty() {
                            match cara.retain(input).await {
                                Ok(memories) => {
                                    if !memories.is_empty() {
                                        tracing::info!("Retained {} new memories", memories.len());
                                    }
                                }
                                Err(e) => tracing::error!("Retain error: {}", e),
                            }

                            match cara.reflect(input, 2000).await {
                                Ok(response) => println!("\n{}\n", response),
                                Err(e) => tracing::error!("Reflect error: {}", e),
                            }
                        }
                    }
                    Err(e) => {
                        if e.kind() != io::ErrorKind::Interrupted {
                            tracing::error!("IO error: {}", e);
                        }
                        break;
                    }
                }
            }
        }
    }

    println!("Goodbye!");
    Ok(())
}
