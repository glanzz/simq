//! Procedural macros for compile-time quantum gate matrix generation
//!
//! This crate provides macros that generate gate matrices at compile time,
//! eliminating runtime computation overhead for frequently-used angles.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, LitFloat, Token};

/// Macro input for generating cached rotation matrices
struct CachedRotationInput {
    gate_type: syn::Ident,
    _comma1: Token![,],
    angles: syn::punctuated::Punctuated<LitFloat, Token![,]>,
}

impl Parse for CachedRotationInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(CachedRotationInput {
            gate_type: input.parse()?,
            _comma1: input.parse()?,
            angles: syn::punctuated::Punctuated::parse_terminated(input)?,
        })
    }
}

/// Generates a compile-time cached rotation gate matrix cache
///
/// # Example
///
/// ```rust,ignore
/// use ferriq_macros::cached_rotations;
///
/// // Generate compile-time cache for common rotation angles
/// cached_rotations!(RX, 0.0, 0.1, 0.2, 0.5, 1.0, std::f64::consts::PI / 2.0);
/// ```
///
/// This generates:
/// - A constant array of matrices for each angle
/// - A lookup function that returns the cached matrix or computes on-demand
/// - Zero runtime overhead for cached angles
#[proc_macro]
pub fn cached_rotations(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as CachedRotationInput);
    let gate_type = &input.gate_type;

    let angles: Vec<f64> = input
        .angles
        .iter()
        .map(|lit| lit.base10_parse::<f64>().expect("Failed to parse angle"))
        .collect();

    let gate_type_str = gate_type.to_string();
    if !["RX", "RY", "RZ"].contains(&gate_type_str.as_str()) {
        panic!("Unsupported gate type: {}. Supported types: RX, RY, RZ", gate_type);
    }

    let compute_on_demand = match gate_type_str.as_str() {
        "RX" => quote! {
            fn compute_matrix(theta: f64) -> [[Complex64; 2]; 2] {
                let half: f64 = theta / 2.0;
                let c: f64 = half.cos();
                let s: f64 = half.sin();
                [
                    [Complex64::new(c, 0.0), Complex64::new(0.0, -s)],
                    [Complex64::new(0.0, -s), Complex64::new(c, 0.0)],
                ]
            }
        },
        "RY" => quote! {
            fn compute_matrix(theta: f64) -> [[Complex64; 2]; 2] {
                let half: f64 = theta / 2.0;
                let c: f64 = half.cos();
                let s: f64 = half.sin();
                [
                    [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
                    [Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
                ]
            }
        },
        "RZ" => quote! {
            fn compute_matrix(theta: f64) -> [[Complex64; 2]; 2] {
                let half: f64 = theta / 2.0;
                let c: f64 = half.cos();
                let s: f64 = half.sin();
                [
                    [Complex64::new(c, -s), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(c, s)],
                ]
            }
        },
        _ => unreachable!(),
    };

    // Generate const declarations for each angle
    let const_declarations: Vec<_> = angles
        .iter()
        .enumerate()
        .map(|(i, &angle)| {
            let const_name = syn::Ident::new(
                &format!("CACHED_{}_{}", gate_type, i),
                proc_macro2::Span::call_site(),
            );

            let half_theta = angle / 2.0;
            let cos_val = half_theta.cos();
            let sin_val = half_theta.sin();
            let neg_sin_val = -sin_val;

            match gate_type.to_string().as_str() {
                "RX" => quote! {
                    const #const_name: [[Complex64; 2]; 2] = [
                        [
                            Complex64::new(#cos_val, 0.0),
                            Complex64::new(0.0, #neg_sin_val),
                        ],
                        [
                            Complex64::new(0.0, #neg_sin_val),
                            Complex64::new(#cos_val, 0.0),
                        ],
                    ];
                },
                "RY" => quote! {
                    const #const_name: [[Complex64; 2]; 2] = [
                        [
                            Complex64::new(#cos_val, 0.0),
                            Complex64::new(#neg_sin_val, 0.0),
                        ],
                        [
                            Complex64::new(#sin_val, 0.0),
                            Complex64::new(#cos_val, 0.0),
                        ],
                    ];
                },
                "RZ" => {
                    let im_neg = -sin_val;
                    let im_pos = sin_val;
                    quote! {
                        const #const_name: [[Complex64; 2]; 2] = [
                            [
                                Complex64::new(#cos_val, #im_neg),
                                Complex64::new(0.0, 0.0),
                            ],
                            [
                                Complex64::new(0.0, 0.0),
                                Complex64::new(#cos_val, #im_pos),
                            ],
                        ];
                    }
                },
                _ => unreachable!(),
            }
        })
        .collect();

    // Generate angle array
    let angle_array: Vec<_> = angles.iter().map(|&a| quote! { #a }).collect();

    // Generate cache array with references to const matrices
    let cache_refs: Vec<_> = (0..angles.len())
        .map(|i| {
            let const_name = syn::Ident::new(
                &format!("CACHED_{}_{}", gate_type, i),
                proc_macro2::Span::call_site(),
            );
            quote! { &#const_name }
        })
        .collect();

    let cache_struct_name =
        syn::Ident::new(&format!("{}Cache", gate_type), proc_macro2::Span::call_site());

    let lookup_fn_name = syn::Ident::new(
        &format!("{}_cached", gate_type.to_string().to_lowercase()),
        proc_macro2::Span::call_site(),
    );

    let expanded = quote! {
        use num_complex::Complex64;

        #(
            #[allow(clippy::approx_constant)]
            #const_declarations
        )*

        /// Compile-time generated cache for rotation matrices
        pub struct #cache_struct_name;

        #[allow(clippy::approx_constant)]
        impl #cache_struct_name {
            /// Cached angles
            const ANGLES: &'static [f64] = &[#(#angle_array),*];

            /// Cached matrices
            const MATRICES: &'static [&'static [[Complex64; 2]; 2]] = &[#(#cache_refs),*];

            #compute_on_demand

            /// Lookup a cached matrix or compute on-demand
            ///
            /// Returns a cached matrix if the angle matches exactly, otherwise computes the matrix.
            /// Uses binary search for O(log n) lookup time.
            #[inline]
            pub fn lookup(theta: f64) -> [[Complex64; 2]; 2] {
                // Binary search for exact angle match
                if let Ok(idx) = Self::ANGLES.binary_search_by(|&probe| {
                    probe.partial_cmp(&theta).unwrap()
                }) {
                    // Exact match - return cached matrix
                    *Self::MATRICES[idx]
                } else {
                    // No match - compute on-demand
                    Self::compute_matrix(theta)
                }
            }

            /// Check if an angle is cached
            #[inline]
            pub fn is_cached(theta: f64) -> bool {
                Self::ANGLES.binary_search_by(|&probe| {
                    probe.partial_cmp(&theta).unwrap()
                }).is_ok()
            }

            /// Get the number of cached angles
            #[inline]
            pub const fn num_cached() -> usize {
                Self::ANGLES.len()
            }
        }

        /// Convenience function for cached lookup
        #[inline]
        pub fn #lookup_fn_name(theta: f64) -> [[Complex64; 2]; 2] {
            #cache_struct_name::lookup(theta)
        }
    };

    TokenStream::from(expanded)
}

/// Generates a compile-time matrix cache for a range of angles
///
/// # Example
///
/// ```rust,ignore
/// use ferriq_macros::cache_rotation_range;
///
/// // Cache RX matrices for angles from 0 to π/2 with 100 steps
/// cache_rotation_range!(RX, 0.0, 1.57079632679, 100);
/// ```
#[proc_macro]
pub fn cache_rotation_range(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as CacheRotationRangeInput);
    let gate_type = &input.gate_type;
    let start = input.start.base10_parse::<f64>().unwrap();
    let end = input.end.base10_parse::<f64>().unwrap();
    let steps = input.steps.base10_parse::<usize>().unwrap();

    // Generate evenly-spaced angles
    let angles: Vec<f64> = (0..steps)
        .map(|i| start + (end - start) * (i as f64) / ((steps - 1) as f64))
        .collect();

    let gate_type_str2 = gate_type.to_string();
    if !["RX", "RY", "RZ"].contains(&gate_type_str2.as_str()) {
        panic!("Unsupported gate type: {}. Supported types: RX, RY, RZ", gate_type);
    }

    let range_compute_on_demand = match gate_type_str2.as_str() {
        "RX" => quote! {
            fn compute_matrix(theta: f64) -> [[Complex64; 2]; 2] {
                let half: f64 = theta / 2.0;
                let c: f64 = half.cos();
                let s: f64 = half.sin();
                [
                    [Complex64::new(c, 0.0), Complex64::new(0.0, -s)],
                    [Complex64::new(0.0, -s), Complex64::new(c, 0.0)],
                ]
            }
        },
        "RY" => quote! {
            fn compute_matrix(theta: f64) -> [[Complex64; 2]; 2] {
                let half: f64 = theta / 2.0;
                let c: f64 = half.cos();
                let s: f64 = half.sin();
                [
                    [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
                    [Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
                ]
            }
        },
        "RZ" => quote! {
            fn compute_matrix(theta: f64) -> [[Complex64; 2]; 2] {
                let half: f64 = theta / 2.0;
                let c: f64 = half.cos();
                let s: f64 = half.sin();
                [
                    [Complex64::new(c, -s), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(c, s)],
                ]
            }
        },
        _ => unreachable!(),
    };

    // Generate const array of matrices
    let matrix_elements: Vec<_> = angles
        .iter()
        .map(|&angle| {
            let half_theta = angle / 2.0;
            let cos_val = half_theta.cos();
            let sin_val = half_theta.sin();
            let neg_sin_val = -sin_val;

            match gate_type.to_string().as_str() {
                "RX" => quote! {
                    [
                        [
                            Complex64::new(#cos_val, 0.0),
                            Complex64::new(0.0, #neg_sin_val),
                        ],
                        [
                            Complex64::new(0.0, #neg_sin_val),
                            Complex64::new(#cos_val, 0.0),
                        ],
                    ]
                },
                "RY" => quote! {
                    [
                        [
                            Complex64::new(#cos_val, 0.0),
                            Complex64::new(#neg_sin_val, 0.0),
                        ],
                        [
                            Complex64::new(#sin_val, 0.0),
                            Complex64::new(#cos_val, 0.0),
                        ],
                    ]
                },
                "RZ" => {
                    let im_neg = -sin_val;
                    let im_pos = sin_val;
                    quote! {
                        [
                            [
                                Complex64::new(#cos_val, #im_neg),
                                Complex64::new(0.0, 0.0),
                            ],
                            [
                                Complex64::new(0.0, 0.0),
                                Complex64::new(#cos_val, #im_pos),
                            ],
                        ]
                    }
                },
                _ => unreachable!(),
            }
        })
        .collect();

    let angle_array: Vec<_> = angles.iter().map(|&a| quote! { #a }).collect();

    let cache_struct_name =
        syn::Ident::new(&format!("{}RangeCache", gate_type), proc_macro2::Span::call_site());

    let lookup_fn_name = syn::Ident::new(
        &format!("{}_range_cached", gate_type.to_string().to_lowercase()),
        proc_macro2::Span::call_site(),
    );

    let expanded = quote! {
        use num_complex::Complex64;

        /// Compile-time generated range cache for rotation matrices
        pub struct #cache_struct_name;

        #[allow(clippy::approx_constant)]
        impl #cache_struct_name {
            /// Cached angles (evenly spaced)
            const ANGLES: [f64; #steps] = [#(#angle_array),*];

            /// Cached matrices
            const MATRICES: [[[Complex64; 2]; 2]; #steps] = [#(#matrix_elements),*];

            /// Angle range configuration
            const START: f64 = #start;
            const END: f64 = #end;
            const STEP: f64 = (#end - #start) / (#steps as f64 - 1.0);

            #range_compute_on_demand

            /// Exact lookup with on-demand fallback
            ///
            /// Returns a cached matrix only when `theta` lands exactly
            /// (within 1e-12) on a cached grid angle; any other angle is
            /// computed on demand at full precision. Gate matrices are never
            /// snapped to the nearest grid point — use
            /// [`lookup_interpolated`](Self::lookup_interpolated) to
            /// explicitly opt into approximate lookups.
            #[inline]
            pub fn lookup(theta: f64) -> [[Complex64; 2]; 2] {
                if (Self::START..=Self::END).contains(&theta) {
                    let index = ((theta - Self::START) / Self::STEP).round() as usize;
                    let index = index.min(#steps - 1);
                    if (theta - Self::ANGLES[index]).abs() < 1e-12 {
                        return Self::MATRICES[index];
                    }
                }

                Self::compute_matrix(theta)
            }

            /// Lookup with linear interpolation for better accuracy
            ///
            /// Interpolates between two nearest cached matrices.
            /// Note: Matrix interpolation is approximate and may not preserve unitarity perfectly.
            #[inline]
            pub fn lookup_interpolated(theta: f64) -> [[Complex64; 2]; 2] {
                if !(Self::START..=Self::END).contains(&theta) {
                    return Self::compute_matrix(theta);
                }

                // Find surrounding indices
                let index_f = (theta - Self::START) / Self::STEP;
                let index = index_f as usize;

                if index >= #steps - 1 {
                    // At or beyond last entry
                    return Self::MATRICES[#steps - 1];
                }

                // Linear interpolation between matrices
                let frac = index_f - index as f64;
                let m1 = &Self::MATRICES[index];
                let m2 = &Self::MATRICES[index + 1];

                [
                    [
                        m1[0][0] * (1.0 - frac) + m2[0][0] * frac,
                        m1[0][1] * (1.0 - frac) + m2[0][1] * frac,
                    ],
                    [
                        m1[1][0] * (1.0 - frac) + m2[1][0] * frac,
                        m1[1][1] * (1.0 - frac) + m2[1][1] * frac,
                    ],
                ]
            }

            /// Get the number of cached matrices
            #[inline]
            pub const fn num_cached() -> usize {
                #steps
            }

            /// Get cache memory usage in bytes
            #[inline]
            pub const fn memory_bytes() -> usize {
                std::mem::size_of::<[[Complex64; 2]; 2]>() * #steps
            }
        }

        /// Convenience function for cached lookup
        #[inline]
        pub fn #lookup_fn_name(theta: f64) -> [[Complex64; 2]; 2] {
            #cache_struct_name::lookup(theta)
        }
    };

    TokenStream::from(expanded)
}

struct CacheRotationRangeInput {
    gate_type: syn::Ident,
    _comma1: Token![,],
    start: LitFloat,
    _comma2: Token![,],
    end: LitFloat,
    _comma3: Token![,],
    steps: syn::LitInt,
}

impl Parse for CacheRotationRangeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(CacheRotationRangeInput {
            gate_type: input.parse()?,
            _comma1: input.parse()?,
            start: input.parse()?,
            _comma2: input.parse()?,
            end: input.parse()?,
            _comma3: input.parse()?,
            steps: input.parse()?,
        })
    }
}
