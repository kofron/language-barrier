use language_barrier_core::{Result, ToolDefinition};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ===== Weather Tool =====

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct WeatherRequest {
    pub location: String,
    pub units: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WeatherResponse {
    pub temperature: i32,
    pub conditions: String,
    pub location: String,
    pub units: String,
}

pub struct WeatherTool;

impl ToolDefinition for WeatherTool {
    type Input = WeatherRequest;
    type Output = WeatherResponse;

    fn name(&self) -> String {
        "get_weather".to_string()
    }

    fn description(&self) -> String {
        "Get current weather for a location".to_string()
    }
}

// ===== Calculator Tool =====

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct CalculatorRequest {
    pub expression: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalculatorResponse {
    pub result: f64,
    pub expression: String,
}

pub struct CalculatorTool;

impl ToolDefinition for CalculatorTool {
    type Input = CalculatorRequest;
    type Output = CalculatorResponse;

    fn name(&self) -> String {
        "calculate".to_string()
    }

    fn description(&self) -> String {
        "Calculate the result of a mathematical expression".to_string()
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
