use std::fmt;

#[derive(Debug, Clone)]
pub enum RuntimeStatusError {
    EXEC_FINISH,
    RT_ERROR,
}

impl fmt::Display for RuntimeStatusError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self))
    }
}
