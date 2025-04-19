use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use language_barrier_core::message::{Content, Function, ToolCall};
use language_barrier_core::{
    Chat, Claude, Message, Result, Tool, ToolCallView, ToolDescription, Toolbox, TypedToolbox,
};

// Define a typed weather tool request
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherRequest {
    location: String,
    units: Option<String>,
}

impl Tool for WeatherRequest {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get current weather for a location"
    }
}

// Define a calculator tool request
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CalculatorRequest {
    expression: String,
}

impl Tool for CalculatorRequest {
    fn name(&self) -> &str {
        "calculate"
    }

    fn description(&self) -> &str {
        "Calculate mathematical expressions"
    }
}

// Define a combined enum for our tool requests
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ExampleToolRequest {
    Weather(WeatherRequest),
    Calculator(CalculatorRequest),
}

// Define our example toolbox that implements both the untyped Toolbox trait
// and the typed TypedToolbox trait
struct ExampleToolbox;

impl Toolbox for ExampleToolbox {
    fn describe(&self) -> Vec<ToolDescription> {
        // Create schema for WeatherRequest
        let weather_schema = schemars::schema_for!(WeatherRequest);
        let weather_schema_value = serde_json::to_value(weather_schema.schema).unwrap();

        // Create schema for CalculatorRequest
        let calculator_schema = schemars::schema_for!(CalculatorRequest);
        let calculator_schema_value = serde_json::to_value(calculator_schema.schema).unwrap();

        vec![
            ToolDescription {
                name: "get_weather".to_string(),
                description: "Get current weather for a location".to_string(),
                parameters: weather_schema_value,
            },
            ToolDescription {
                name: "calculate".to_string(),
                description: "Calculate mathematical expressions".to_string(),
                parameters: calculator_schema_value,
            },
        ]
    }

    fn execute(&self, name: &str, arguments: Value) -> Result<String> {
        match name {
            "get_weather" => {
                let request: WeatherRequest = serde_json::from_value(arguments)?;
                let units = request.units.unwrap_or_else(|| "celsius".to_string());
                Ok(format!(
                    "Weather in {}: 22 degrees {}, partly cloudy",
                    request.location, units
                ))
            }
            "calculate" => {
                let request: CalculatorRequest = serde_json::from_value(arguments)?;
                // This is a simple example - in a real implementation, you'd evaluate the expression
                Ok(format!("Result: {}", request.expression))
            }
            _ => Err(language_barrier_core::Error::ToolNotFound(name.to_string())),
        }
    }
}

impl TypedToolbox<ExampleToolRequest> for ExampleToolbox {
    fn parse_tool_call(&self, tool_call: &ToolCall) -> Result<ExampleToolRequest> {
        let name = &tool_call.function.name;
        let arguments = serde_json::from_str(&tool_call.function.arguments)?;

        match name.as_str() {
            "get_weather" => {
                let request: WeatherRequest = serde_json::from_value(arguments)?;
                Ok(ExampleToolRequest::Weather(request))
            }
            "calculate" => {
                let request: CalculatorRequest = serde_json::from_value(arguments)?;
                Ok(ExampleToolRequest::Calculator(request))
            }
            _ => Err(language_barrier_core::Error::ToolNotFound(name.clone())),
        }
    }

    fn execute_typed(&self, request: ExampleToolRequest) -> Result<String> {
        match request {
            ExampleToolRequest::Weather(weather_req) => {
                let units = weather_req.units.unwrap_or_else(|| "celsius".to_string());
                Ok(format!(
                    "Weather in {}: 22 degrees {}, partly cloudy",
                    weather_req.location, units
                ))
            }
            ExampleToolRequest::Calculator(calc_req) => {
                // This is a simple example - in a real implementation, you'd evaluate the expression
                Ok(format!("Result: {}", calc_req.expression))
            }
        }
    }
}

fn main() -> Result<()> {
    // Create a new chat with Claude
    let mut chat = Chat::new(Claude::Haiku3)
        .with_system_prompt("You are a helpful assistant with access to tools.")
        .with_toolbox(ExampleToolbox);

    // Add example messages
    chat.add_message(Message::user("What's the weather in San Francisco?"));

    // In a real scenario, the assistant would generate this message with tool calls
    // Here we manually create it for demonstration purposes

    // Create a tool call
    let tool_call = ToolCall {
        id: "call_abc123".to_string(),
        tool_type: "function".to_string(),
        function: Function {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"San Francisco","units":"fahrenheit"}"#.to_string(),
        },
    };

    // Create new assistant message with the tool call
    let assistant_message = Message::assistant_with_tool_calls(vec![tool_call]);

    // Add the message to chat history
    chat.add_message(assistant_message.clone());

    // Process the tool calls (this would typically be done by the executor in real usage)
    chat.process_tool_calls(&assistant_message)?;

    // Now, demonstrate the typed view approach
    let toolbox = ExampleToolbox;
    let view = ToolCallView::<ExampleToolRequest, _>::new(&toolbox, &assistant_message);

    if view.has_tool_calls() {
        let tool_requests = view.tool_calls()?;

        for request in tool_requests {
            // Type-safe pattern matching on the tool request
            match request {
                ExampleToolRequest::Weather(weather_req) => {
                    println!(
                        "Found weather request for location: {} with units: {:?}",
                        weather_req.location, weather_req.units
                    );

                    // Execute the request using the typed method
                    let result = toolbox.execute_typed(ExampleToolRequest::Weather(weather_req))?;
                    println!("Result: {}", result);
                }
                ExampleToolRequest::Calculator(calc_req) => {
                    println!("Found calculator request: {}", calc_req.expression);

                    // Execute the request using the typed method
                    let result = toolbox.execute_typed(ExampleToolRequest::Calculator(calc_req))?;
                    println!("Result: {}", result);
                }
            }
        }
    }

    // Print the full conversation
    println!("\nFull conversation:");
    for msg in chat.history.iter() {
        match msg {
            Message::User { content, .. } => {
                if let Content::Text(text) = content {
                    println!("User: {}", text);
                } else {
                    println!("User: {:?}", content);
                }
            }
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                if let Some(Content::Text(text)) = content {
                    println!("Assistant: {}", text);
                } else if let Some(content) = content {
                    println!("Assistant: {:?}", content);
                } else {
                    println!("Assistant: [no content]");
                }

                if !tool_calls.is_empty() {
                    for call in tool_calls {
                        println!(
                            "  Tool Call: {} ({}) - {}",
                            call.id, call.function.name, call.function.arguments
                        );
                    }
                }
            }
            Message::Tool {
                tool_call_id,
                content,
                ..
            } => {
                println!("Tool ({}): {}", tool_call_id, content);
            }
            Message::System { content, .. } => {
                println!("System: {}", content);
            }
        }
    }

    Ok(())
}
