//! jieba-rs 0.11.0 (the version rjieba 0.2.2 embeds) for the browser, as a plain C ABI with no JS glue.
//! JS writes UTF-8 into `alloc`'d memory, calls `cut`, then reads `cut`'s count of u32 token end offsets
//! (bytes into the input) from `result_ptr`. The dictionary is jieba-rs's own dict.txt, embedded.

use jieba_rs::Jieba;
use std::sync::OnceLock;

static DICT: &str = include_str!(env!("JIEBA_DICT"));
static JIEBA: OnceLock<Jieba> = OnceLock::new();
static mut RESULT: Vec<u32> = Vec::new();

fn jieba() -> &'static Jieba {
    // dict.txt has no duplicate words, so with_dict builds what Jieba::new() would
    JIEBA.get_or_init(|| Jieba::with_dict(&mut DICT.as_bytes()).expect("dict.txt"))
}

/// Builds the dictionary now rather than on the first `cut`.
#[no_mangle]
pub extern "C" fn init() {
    jieba();
}

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// # Safety
/// `ptr` and `len` come from one `alloc` call.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len));
}

/// Segments `len` bytes of UTF-8 at `ptr`, as rjieba.cut(text, hmm) does. Returns the token count, or
/// u32::MAX if the bytes are not UTF-8.
///
/// # Safety
/// `ptr` points to `len` initialised bytes.
#[no_mangle]
pub unsafe extern "C" fn cut(ptr: *const u8, len: usize, hmm: bool) -> u32 {
    let Ok(text) = std::str::from_utf8(std::slice::from_raw_parts(ptr, len)) else {
        return u32::MAX;
    };
    let ends: Vec<u32> = jieba().cut(text, hmm).iter().map(|t| t.byte_end as u32).collect();
    let n = ends.len() as u32;
    RESULT = ends;
    n
}

/// Where the last `cut` left its token end offsets.
#[no_mangle]
pub extern "C" fn result_ptr() -> *const u32 {
    // SAFETY: single-threaded wasm; read only between calls to cut
    unsafe { (*std::ptr::addr_of!(RESULT)).as_ptr() }
}
