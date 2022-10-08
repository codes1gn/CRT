// external crates
use nom::types::CompleteStr;
use nom::*;

use super::assembler_base::*;
use super::parse_literal::parse_integer_list;
use super::parse_type::*;

// "crt.helper.zeros<[2 3]>"
named!(pub parse_helper_zeros<CompleteStr, Token>,
    do_parse!(
        _sp: space0 >>
        tag!("zeros<") >>
        _shape: parse_integer_list >>
        tag!(">") >>
        (
            Token::UninitTensor { data_generator: 0f32, shape: _shape }
        )
    )
);

// "crt.helper.ones<[2 3]>"
named!(pub parse_helper_ones<CompleteStr, Token>,
    do_parse!(
        _sp: space0 >>
        tag!("ones<") >>
        _shape: parse_integer_list >>
        tag!(">") >>
        (
            Token::UninitTensor { data_generator: 1f32, shape: _shape }
        )
    )
);

// "crt.helper.uniform<[2 3]>"
named!(pub parse_helper_uniform<CompleteStr, Token>,
    do_parse!(
        _sp: space0 >>
        tag!("uniform<") >>
        _shape: parse_integer_list >>
        tag!(">") >>
        (
            Token::UninitRNGTensor { distribution: 0 as u8, shape: _shape }
        )
    )
);

// "crt.helper.normal<[2 3]>"
named!(pub parse_helper_normal<CompleteStr, Token>,
    do_parse!(
        _sp: space0 >>
        tag!("normal<") >>
        _shape: parse_integer_list >>
        tag!(">") >>
        (
            Token::UninitRNGTensor { distribution: 1 as u8, shape: _shape }
        )
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_helper_zeros() {
        // w.o. \n
        let result = parse_helper_zeros(CompleteStr("zeros<[2 3 1]>"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::UninitTensor {
                data_generator: 0f32,
                shape: vec![2, 3, 1]
            }
        );
    }

    #[test]
    fn test_parse_helper_ones() {
        // w.o. \n
        let result = parse_helper_ones(CompleteStr("ones<[2 3 1]>"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::UninitTensor {
                data_generator: 1f32,
                shape: vec![2, 3, 1]
            }
        );
    }

    #[test]
    fn test_parse_helper_uniform() {
        // w.o. \n
        let result = parse_helper_uniform(CompleteStr("uniform<[2 3 1]>"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::UninitRNGTensor {
                distribution: 0 as u8,
                shape: vec![2, 3, 1]
            }
        );
    }

    #[test]
    fn test_parse_helper_normal() {
        // w.o. \n
        let result = parse_helper_normal(CompleteStr("normal<[2 3 1]>"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::UninitRNGTensor {
                distribution: 1 as u8,
                shape: vec![2, 3, 1]
            }
        );
    }
}
