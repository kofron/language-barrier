use std::sync::{Arc, Mutex};

use language_barrier::{
    Chat, Message, Mistral, Secret, SingleRequestExecutor, Tool, ToolDescription, Toolbox,
};
use language_barrier::message::{Content, ContentPart, Function, ToolCall};
use language_barrier::provider::mistral::{MistralConfig, MistralProvider};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CalculatorRequest {
    operation: String,
    a: f64,
    b: f64,
}

impl Tool for CalculatorRequest {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform a calculation between two numbers"
    }
}

#[derive(Clone)]
struct CalculatorToolbox {
    call_count: Arc<Mutex<usize>>,
}

impl CalculatorToolbox {
    fn new() -> Self {
        Self {
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

impl Toolbox for CalculatorToolbox {
    fn describe(&self) -> Vec<ToolDescription> {
        let schema = schemars::schema_for!(CalculatorRequest);
        let schema_value = serde_json::to_value(schema.schema).unwrap();

        vec![ToolDescription {
            name: "calculator".to_string(),
            description: "Perform a calculation between two numbers".to_string(),
            parameters: schema_value,
        }]
    }

    fn execute(&self, name: &str, arguments: Value) -> language_barrier::Result<String> {
        if name != "calculator" {
            return Err(language_barrier::Error::ToolNotFound(name.to_string()));
        }

        // Increment call count
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        // Parse the arguments
        let request: CalculatorRequest = serde_json::from_value(arguments)?;
        
        // Perform the calculation
        let result = match request.operation.as_str() {
            "add" => request.a + request.b,
            "subtract" => request.a - request.b,
            "multiply" => request.a * request.b,
            "divide" => {
                if request.b == 0.0 {
                    return Err(language_barrier::Error::ToolExecutionError(
                        "Division by zero".to_string(),
                    ));
                }
                request.a / request.b
            }
            _ => {
                return Err(language_barrier::Error::ToolExecutionError(format!(
                    "Unsupported operation: {}",
                    request.operation
                )))
            }
        };

        Ok(format!("The result of {} {} {} is {}", 
            request.a, request.operation, request.b, result))
    }
}

#[tokio::test]
async fn test_mistral_tools() {
    // Skip the test if MISTRAL_API_KEY is not set
    let api_key = match std::env::var("MISTRAL_API_KEY") {
        Ok(key) if !key.is_empty() => Secret::new(key),
        _ => {
            println!("Skipping test_mistral_tools: No MISTRAL_API_KEY found");
            return;
        }
    };

    // Create a provider with the API key
    let config = MistralConfig {
        api_key: api_key.inner().to_string(),
        base_url: "https://api.mistral.ai/v1".to_string(),
    };
    let provider = MistralProvider::with_config(config);
    
    // Create an executor with our provider
    let executor = SingleRequestExecutor::new(provider);
    
    // Create a toolbox to track tool calls
    let toolbox = CalculatorToolbox::new();
    
    // Create a chat with the Mistral Small model
    let mut chat = Chat::new(Mistral::Small)
        .with_system_prompt("You are a helpful AI assistant that uses tools when appropriate. Always use the calculator tool for math problems.")
        .with_max_output_tokens(256)
        .with_toolbox(toolbox.clone());
    
    // Add a user message that should trigger the calculator tool
    chat.add_message(Message::user("What is 123 multiplied by 456?"));
    
    // Create a new mutable chat for each request
    // Send the chat to get a response
    let response = executor.send(chat).await.unwrap();
    
    // Check if the response includes tool calls
    let has_tool_calls = match &response {
        Message::Assistant { tool_calls, .. } => !tool_calls.is_empty(),
        _ => false,
    };
    
    if has_tool_calls {
        println!("Tool calls detected, processing...");
        
        // Create a new chat with the same configuration
        let mut new_chat = Chat::new(Mistral::Small)
            .with_system_prompt("You are a helpful AI assistant that uses tools when appropriate. Always use the calculator tool for math problems.")
            .with_max_output_tokens(256)
            .with_toolbox(toolbox.clone());
            
        // Add the original user message
        new_chat.add_message(Message::user("What is 123 multiplied by 456?"));
        
        // Add the assistant response with tool calls
        new_chat.add_message(response.clone());
        
        // Process the tool calls
        new_chat.process_tool_calls(&response).unwrap();
        
        // Send another message to get the final answer
        let final_response = executor.send(new_chat).await.unwrap();
        
        // Verify that the tool was called
        assert_eq!(toolbox.get_call_count(), 1);
        
        // Check that the final response contains the correct result
        match &final_response {
            Message::Assistant { content, .. } => {
                match content {
                    Some(Content::Text(text)) => {
                        println!("Final response: {}", text);
                        assert!(text.contains("56088"));
                    },
                    _ => {
                        println!("Unexpected content format");
                        assert!(false, "Expected text content");
                    }
                }
            },
            _ => assert!(false, "Expected assistant message"),
        }
    } else {
        // Some models might get the calculation right without using tools
        println!("No tool calls, checking if response is still correct...");
        
        match &response {
            Message::Assistant { content, .. } => {
                match content {
                    Some(Content::Text(text)) => {
                        println!("Response: {}", text);
                        
                        // Check if the response contains the correct result even without tool use
                        assert!(
                            text.contains("56088") || 
                            text.contains("56,088") || 
                            text.contains("123 × 456 = 56088")
                        );
                    },
                    _ => {
                        println!("Unexpected content format");
                        assert!(false, "Expected text content");
                    }
                }
            },
            _ => assert!(false, "Expected assistant message"),
        }
    }
}