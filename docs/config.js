// Where the demo fetches the bitstreams (*.wcdc) and tokenizer.json from.
//  * relative "bitstreams/" = same site: the Hugging Face static Space and local previews
//    (phase2/demo_build.py stages them into docs/bitstreams/, which is git-ignored);
//  * GitHub Pages cannot serve them from git (they are not committed) -> point dataBase to a public host
//    that sends CORS headers, e.g. the public Space or a public HF dataset/model repo:
//      "https://huggingface.co/spaces/<user>/<space>/resolve/main/bitstreams/"
//      "https://huggingface.co/datasets/<user>/<repo>/resolve/main/"
//  * a "?data=<url>" query parameter overrides this for one visit.
window.WC_CONFIG = {
  // GitHub Pages serves this page but not the (git-ignored) bitstreams, so fetch them from the
  // public Hugging Face Space, which hosts the same files and sends CORS headers.
  // Append "?data=bitstreams/" to the URL to use same-site data instead (e.g. on the Space itself).
  dataBase: "https://huggingface.co/spaces/lifeart/weight-codec/resolve/main/bitstreams/",
};
