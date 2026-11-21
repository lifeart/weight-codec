// Where the demo fetches the bitstreams (*.wcdc) and tokenizer.json from.
//  * relative "bitstreams/" = same site: the Hugging Face static Space and local previews
//    (phase2/demo_build.py stages them into docs/bitstreams/, which is git-ignored);
//  * GitHub Pages cannot serve them from git (they are not committed) -> point dataBase to a public host
//    that sends CORS headers, e.g. the public Space or a public HF dataset/model repo:
//      "https://huggingface.co/spaces/<user>/<space>/resolve/main/bitstreams/"
//      "https://huggingface.co/datasets/<user>/<repo>/resolve/main/"
//  * a "?data=<url>" query parameter overrides this for one visit.
window.WC_CONFIG = {
  dataBase: "bitstreams/",
};
