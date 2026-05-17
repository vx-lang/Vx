use crate::instantiate_hash_map_ffi;

// Common hash map specializations
instantiate_hash_map_ffi!(i32_i32, i32, i32);
instantiate_hash_map_ffi!(i32_f32, i32, f32);
