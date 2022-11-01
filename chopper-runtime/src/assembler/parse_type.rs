// external crates
use nom::types::CompleteStr;
use nom::*;

use crate::assembler::assembler_base::*;
use crate::assembler::parse_literal::*;
use crate::base::*;

// (tensor<2x3xf32>, tensor<1x1x28x28xf32>) -> tensor<1x28x28xf32>
// ANCHOR currently only encode return type
named!(pub parse_function_type<CompleteStr, Token>,
    do_parse!(
        tag!("(") >>
        parse_some_ranked_tensor_type >>
        tag!(") -> ") >>
        ret_type: parse_ranked_tensor_type >>
        (
            ret_type
        )
    )
);

// tensor<2x3xf32>, tensor<1x1x28x28xf32>
named!(pub parse_some_ranked_tensor_type<CompleteStr, Vec<Token>>,
    do_parse!(
        tensors: many0!(parse_ranked_tensor_type) >>
        (
            tensors
        )
    )
);

// tensor<1x1x28x28xf32>
named!(pub parse_ranked_tensor_type<CompleteStr, Token>,
    do_parse!(
        space0 >>
        tag!("tensor<") >>
        ranked_shape: parse_ranked_tensor_shape >>
        dtype_token: alt!(
            tag!("i32") | tag!("f32") | tag!("i64")
        ) >>
        tag!(">") >>
        opt!(tag!(",")) >>
        space0 >>
        (
            Token::TensorType {
                dtype: ElementType::from(dtype_token),
                shape: ranked_shape,
            }
        )
    )
);

named!(pub parse_type<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        token: alt!(
            tag!("i32") | tag!("f32")
        ) >>
        ( Token::DType { element_type: ElementType::from(token) } )
    )
);

// TODO hardcode for temp, make it into combined type indicator and type annotation.
named!(pub parse_i32_type<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        tag!(": i32") >>
        ( Token::DType { element_type: ElementType::I32 } )
    )
);

named!(pub parse_f32_type<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        tag!(": f32") >>
        ( Token::DType { element_type: ElementType::F32 } )
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_type() {
        // w.o. \n
        let result = parse_type(CompleteStr("i32"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::DType {
                element_type: ElementType::I32
            }
        );

        // w. \n
        let result = parse_type(CompleteStr(" f32\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::DType {
                element_type: ElementType::F32
            }
        );
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_parse_tensor_type() {
        // w.o. \n
        let result = parse_ranked_tensor_type(CompleteStr("tensor<1x1x28x28xf32>"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::TensorType {
                dtype: ElementType::F32,
                shape: vec![1, 1, 28, 28],
            }
        );
        let result = parse_ranked_tensor_type(CompleteStr("tensor<1x1x28x28xf32>,"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::TensorType {
                dtype: ElementType::F32,
                shape: vec![1, 1, 28, 28],
            }
        );
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_parse_some_tensor_type() {
        // w.o. \n
        let result = parse_some_ranked_tensor_type(CompleteStr(
            "tensor<1x1x28x28xf32>, tensor<1x1x28x28xf32>",
        ));
        assert_eq!(result.is_ok(), true);
        let (_remain, _bytes_result) = result.unwrap();
        assert_eq!(_remain, CompleteStr(""));
        assert_eq!(
            _bytes_result,
            vec![
                Token::TensorType {
                    dtype: ElementType::F32,
                    shape: vec![1, 1, 28, 28],
                },
                Token::TensorType {
                    dtype: ElementType::F32,
                    shape: vec![1, 1, 28, 28],
                },
            ]
        );
    }

    #[cfg(feature = "mock")]
    #[test]
    fn test_parse_function_type() {
        // w.o. \n
        let result = parse_function_type(CompleteStr(
            "(tensor<2x3xf32>, tensor<1x1x28x28xf32>) -> tensor<1x28x28xf32>",
        ));
        assert_eq!(result.is_ok(), true);
        let (_remain, _bytes_result) = result.unwrap();
        assert_eq!(_remain, CompleteStr(""));
        assert_eq!(
            _bytes_result,
            Token::TensorType {
                dtype: ElementType::F32,
                shape: vec![1, 28, 28],
            },
        );
    }
}
