use core::mem;

use proc_macro2::{Ident, Span};

use quote::quote;

use syn::punctuated::Punctuated;
use syn::token::{Bracket, Comma, Paren};
use syn::{parse_macro_input, parse_quote, DeriveInput};
use syn::{Expr, Visibility};
use syn::{ExprArray, ExprCall, ItemConst};
use syn::{ExprLit, IntSuffix, Lit, LitInt};

const NUM_CLASSES: usize = 83;
// These size classes were calculated based on the optimal
// amount of wasted space per class when blocks are stored
// in the same memory region as SlabCarrier + AtomicBlockSet.
// All of these classes have at most 88 bytes of wasted space
// in a region of 262k, which is the size of a super-aligned
// SlabCarrier.
const SIZE_CLASSES: [usize; NUM_CLASSES] = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 17, 22, 24, 29, 34, 37, 50, 89, 92, 103, 106, 159,
    179, 183, 206, 260, 273, 280, 309, 312, 315, 318, 360, 364, 390, 420, 431, 455, 468, 504, 520,
    537, 546, 585, 618, 630, 697, 728, 780, 799, 819, 840, 862, 910, 936, 1092, 1170, 1260, 1365,
    1489, 1560, 1638, 1724, 1820, 1927, 2184, 2340, 2520, 2730, 2978, 3276, 3640, 4095, 4680, 5459,
    5460, 6551, 6552, 8189, 8190,
];

pub(crate) fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Parse into syntax tree
    let derive = parse_macro_input!(input as DeriveInput);
    // Structure name
    let name = derive.ident;
    // Metadata
    let max_size_class = SIZE_CLASSES[NUM_CLASSES - 1];
    // Size classes
    let mut size_class_elems: Punctuated<Expr, Comma> = Punctuated::new();
    for item in SIZE_CLASSES.iter() {
        let mut args: Punctuated<Expr, Comma> = Punctuated::new();
        args.push_value(Expr::Lit(ExprLit {
            attrs: Vec::new(),
            lit: Lit::Int(LitInt::new(
                *item as u64,
                IntSuffix::None,
                Span::call_site(),
            )),
        }));
        args.push_punct(parse_quote!(,));
        size_class_elems.push_value(Expr::Call(ExprCall {
            attrs: Vec::new(),
            func: Box::new(parse_quote!(
                liblumen_core::alloc::size_classes::SizeClass::new
            )),
            paren_token: Paren {
                span: Span::call_site(),
            },
            args,
        }));
        size_class_elems.push_punct(parse_quote!(,));
    }
    let size_class_len = NUM_CLASSES;
    let size_class_const = ItemConst {
        attrs: Vec::new(),
        vis: parse_quote!(pub),
        const_token: parse_quote!(const),
        ident: Ident::new("SIZE_CLASSES", Span::call_site()),
        colon_token: parse_quote!(:),
        ty: Box::new(
            parse_quote!([liblumen_core::alloc::size_classes::SizeClass; #size_class_len]),
        ),
        eq_token: parse_quote!(=),
        expr: Box::new(Expr::Array(ExprArray {
            attrs: Vec::new(),
            bracket_token: Bracket {
                span: Span::call_site(),
            },
            elems: size_class_elems,
        })),
        semi_token: parse_quote!(;),
    };
    // Word size to size class index
    let words_to_class_index = generate_words_to_class_index(&SIZE_CLASSES);
    let words_to_class_index_len = words_to_class_index.len();
    let mut words_to_class_index_elems: Punctuated<Expr, Comma> = Punctuated::new();
    for item in words_to_class_index {
        words_to_class_index_elems.push_value(Expr::Lit(ExprLit {
            attrs: Vec::new(),
            lit: Lit::Int(LitInt::new(item as u64, IntSuffix::None, Span::call_site())),
        }));
        words_to_class_index_elems.push_punct(parse_quote!(,));
    }
    let words_to_class_const = ItemConst {
        attrs: Vec::new(),
        vis: Visibility::Inherited,
        const_token: parse_quote!(const),
        ident: Ident::new("WORDS_TO_CLASS", Span::call_site()),
        colon_token: parse_quote!(:),
        ty: Box::new(parse_quote!([usize; #words_to_class_index_len])),
        eq_token: parse_quote!(=),
        expr: Box::new(Expr::Array(ExprArray {
            attrs: Vec::new(),
            bracket_token: Bracket {
                span: Span::call_site(),
            },
            elems: words_to_class_index_elems,
        })),
        semi_token: parse_quote!(;),
    };
    let size_class_path: syn::Path = parse_quote!(liblumen_core::alloc::size_classes::SizeClass);
    // Build ouput syntax tree
    let output = quote! {
        impl #name {
            // The size of a word on the target
            const __WORD_SIZE: usize = core::mem::size_of::<usize>();
            /// The number of size classes
            pub const NUM_SIZE_CLASSES: usize = #NUM_CLASSES;
            /// The largest size class (in bytes)
            pub const MAX_SIZE_CLASS: #size_class_path = #size_class_path::new(#max_size_class);
            /// The available size classes, each class is the size in words of that class
            #size_class_const
            // Maps allocation request sizes to size class
            #words_to_class_const
        }
        impl liblumen_core::alloc::size_classes::SizeClassIndex for #name {
            #[inline]
            fn index_for(&self, size_class: SizeClass) -> usize {
                use liblumen_core::alloc::size_classes;
                let num_words = size_classes::next_factor_of_word(size_class.to_bytes());
                Self::WORDS_TO_CLASS[num_words]
            }

            #[inline]
            fn size_class_for(&self, request_size: usize) -> Option<#size_class_path> {
                use liblumen_core::alloc::size_classes;

                let num_words = size_classes::next_factor_of_word(request_size);
                if num_words > Self::MAX_SIZE_CLASS.as_words() {
                    return None;
                }
                Some(Self::SIZE_CLASSES[Self::WORDS_TO_CLASS[num_words]])
            }

            #[inline]
            unsafe fn size_class_for_unchecked(&self, request_size: usize) -> #size_class_path {
                use liblumen_core::alloc::size_classes;

                let num_words = size_classes::next_factor_of_word(request_size);
                Self::SIZE_CLASSES[Self::WORDS_TO_CLASS[num_words]]
            }
        }
    };

    proc_macro::TokenStream::from(output)
}

fn generate_words_to_class_index(classes: &[usize]) -> Vec<usize> {
    let mut index = Vec::new();
    // Track request size, which will be assumed to be rounded up
    // to the next factor of the target word size, e.g. with word size
    // of 8, a request of 17 would be rounded up to 24
    let mut num_words: usize = 0;
    for (i, class) in classes.iter().enumerate() {
        while num_words <= *class {
            index.push(i);
            num_words = next_factor_of_word((num_words + 1) * 8);
        }
    }

    index
}

#[inline]
fn next_factor_of_word(n: usize) -> usize {
    let base = n / mem::size_of::<usize>();
    let rem = n % mem::size_of::<usize>();
    if rem == 0 {
        base
    } else {
        base + 1
    }
}
