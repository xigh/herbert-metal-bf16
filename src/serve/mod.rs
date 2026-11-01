//! herbert-metal-serve: the engine behind llama-server's HTTP API (the OpenAI
//! chat completions, `/props`, `/apply-template`, `/tokenize`), for clients
//! written against llama-server.

pub mod output;
pub mod prefix_cache;
pub mod template;
pub mod worker;
