use language_barrier_core::{Result, Tool, ToolDescription, Toolbox};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ===== Weather Tool =====

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WeatherRequest {
    pub location: String,
    pub units: Option<String>,
}

impl Tool for WeatherRequest {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get current weather for a location"
    }
}

// ===== Calculator Tool =====

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalculatorRequest {
    pub expression: String,
}

impl Tool for CalculatorRequest {
    fn name(&self) -> &str {
        "calculate"
    }

    fn description(&self) -> &str {
        "Calculate the result of a mathematical expression"
    }
}

// ===== Common Toolbox Implementation =====

pub struct TestToolbox;

impl Toolbox for TestToolbox {
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
                description: "Calculate the result of a mathematical expression".to_string(),
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
                    "Weather in {}: 22 degrees {}, partly cloudy with a chance of rain",
                    request.location, units
                ))
            }
            "calculate" => {
                let request: CalculatorRequest = serde_json::from_value(arguments)?;
                // Simple calculator that supports basic operations
                let result = match simple_eval(&request.expression) {
                    Ok(value) => format!("The result of {} is {}", request.expression, value),
                    Err(err) => format!("Error calculating {}: {}", request.expression, err),
                };
                Ok(result)
            }
            _ => Err(language_barrier_core::Error::ToolNotFound(name.to_string())),
        }
    }
}

// Simple expression evaluator for calculator tool
fn simple_eval(expr: &str) -> Result<f64> {
    let expr = expr.trim();

    if expr.contains('+') {
        let parts: Vec<&str> = expr.split('+').collect();
        if parts.len() == 2 {
            let a = simple_eval(parts[0])?;
            let b = simple_eval(parts[1])?;
            return Ok(a + b);
        }
    } else if expr.contains('-') {
        let parts: Vec<&str> = expr.split('-').collect();
        if parts.len() == 2 {
            let a = simple_eval(parts[0])?;
            let b = simple_eval(parts[1])?;
            return Ok(a - b);
        }
    } else if expr.contains('*') {
        let parts: Vec<&str> = expr.split('*').collect();
        if parts.len() == 2 {
            let a = simple_eval(parts[0])?;
            let b = simple_eval(parts[1])?;
            return Ok(a * b);
        }
    } else if expr.contains('/') {
        let parts: Vec<&str> = expr.split('/').collect();
        if parts.len() == 2 {
            let a = simple_eval(parts[0])?;
            let b = simple_eval(parts[1])?;
            if b == 0.0 {
                return Err(language_barrier_core::Error::Other(
                    "Division by zero".into(),
                ));
            }
            return Ok(a / b);
        }
    }

    // If no operators, try to parse as a number
    match expr.trim().parse::<f64>() {
        Ok(num) => Ok(num),
        Err(_) => Err(language_barrier_core::Error::Other(format!(
            "Invalid expression: {}",
            expr
        ))),
    }
}

// ===== Weather-only Toolbox =====

pub struct WeatherToolbox;

impl Toolbox for WeatherToolbox {
    fn describe(&self) -> Vec<ToolDescription> {
        // Create schema for WeatherRequest
        let weather_schema = schemars::schema_for!(WeatherRequest);
        let weather_schema_value = serde_json::to_value(weather_schema.schema).unwrap();

        vec![ToolDescription {
            name: "get_weather".to_string(),
            description: "Get current weather for a location".to_string(),
            parameters: weather_schema_value,
        }]
    }

    fn execute(&self, name: &str, arguments: Value) -> Result<String> {
        match name {
            "get_weather" => {
                let request: WeatherRequest = serde_json::from_value(arguments)?;
                let units = request.units.unwrap_or_else(|| "celsius".to_string());
                Ok(format!(
                    "Weather in {}: 22 degrees {}, partly cloudy with a chance of rain",
                    request.location, units
                ))
            }
            _ => Err(language_barrier_core::Error::ToolNotFound(name.to_string())),
        }
    }
}
