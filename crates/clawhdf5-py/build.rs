fn main() {
    // On macOS, the system Python from CommandLineTools may have its shared
    // library and framework in paths that pyo3 doesn't automatically find.
    if cfg!(target_os = "macos") {
        let clt_lib = "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib";
        let clt_fw = "/Library/Developer/CommandLineTools/Library/Frameworks";
        if std::path::Path::new(clt_lib).exists() {
            println!("cargo:rustc-link-search=native={clt_lib}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{clt_fw}");
        }
    }
}
