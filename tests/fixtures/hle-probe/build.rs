fn main() {
    println!("cargo:rustc-link-arg-cdylib=/DELAYLOAD:g6-probe-delayed.dll");
    println!("cargo:rustc-link-lib=delayimp");
}
