//! The model's own chat template (`chat_template.jinja`, else `chat_template`
//! of `tokenizer_config.json`), rendered by minijinja with Python's string and
//! dict methods (`.get`, `.split`): the prompt Google wrote, tools, thinking
//! and tool responses included, as llama-server renders it with `--jinja`.

use crate::core::error::{HerbertError, Result};
use minijinja::{Environment, Value as JValue};
use serde_json::Value;
use std::path::Path;

pub struct Template {
    env: Environment<'static>,
    pub source: String,
}

impl Template {
    pub fn load(model_dir: &Path) -> Result<Template> {
        let source = match std::fs::read_to_string(model_dir.join("chat_template.jinja")) {
            Ok(s) => s,
            Err(_) => {
                let cfg: Value = serde_json::from_slice(&std::fs::read(model_dir.join("tokenizer_config.json"))?)?;
                cfg["chat_template"].as_str().ok_or_else(|| HerbertError::ModelLoad("no chat template in the model directory".into()))?.to_string()
            }
        };
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned("chat", source.clone()).map_err(|e| HerbertError::ModelLoad(format!("chat template: {e:#}")))?;
        Ok(Template { env, source })
    }

    /// The prompt of `messages` (OpenAI format) with `tools`. Tool call
    /// arguments given as a JSON string are parsed first, as llama-server does:
    /// the template writes an object's keys and values in Gemma's own syntax.
    pub fn render(&self, messages: &Value, tools: Option<&Value>, enable_thinking: bool, add_generation_prompt: bool) -> Result<String> {
        let mut messages = messages.clone();
        if let Some(list) = messages.as_array_mut() {
            for m in list {
                if let Some(calls) = m.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    for c in calls {
                        if let Some(args) = c.pointer_mut("/function/arguments") {
                            if let Some(s) = args.as_str() {
                                if let Ok(v) = serde_json::from_str::<Value>(s) {
                                    *args = v;
                                }
                            }
                        }
                    }
                }
            }
        }
        let tools = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
        let ctx = minijinja::context! {
            messages => JValue::from_serialize(&messages),
            tools => tools.map(JValue::from_serialize),
            bos_token => "<bos>",
            eos_token => "<eos>",
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
        };
        let t = self.env.get_template("chat").map_err(|e| HerbertError::Backend(format!("chat template: {e:#}")))?;
        t.render(ctx).map_err(|e| HerbertError::Backend(format!("chat template: {e:#}")))
    }
}
