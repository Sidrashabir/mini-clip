"""
Web interface for Mini-CLIP. Built by build_ui() and served by server.py.

Tabs: Search by text · Search by image · Zero-shot classifier · Embedding map · About & API
"""
import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from inference import Engine

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402

DEFAULT_QUERIES = ["a dog running through the snow", "children playing on the beach",
                   "a man climbing a rock wall", "a person riding a bike on a dirt road",
                   "two dogs playing with a ball in the grass", "a crowd of people at a concert"]
LABEL_SETS = ["dog, cat, horse, bird", "beach, mountain, city street, forest, snow",
              "running, swimming, climbing, cycling, sitting"]

THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#hero h1 { font-size: 2rem; margin-bottom: .2rem; }
.small-note { font-size: .85rem; opacity: .75; }
"""


def overlay_heatmap(path, heat, crop_pct):
    """Draw the occlusion heatmap over the centre crop the model actually saw."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    side = int(min(w, h) * crop_pct)
    left, top = (w - side) // 2, (h - side) // 2
    crop = im.crop((left, top, left + side, top + side)).resize((384, 384))
    heat = np.array(Image.fromarray((np.array(heat) * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR))
    colored = (colormaps["inferno"](heat / 255.0)[..., :3] * 255).astype(np.uint8)
    alpha = (heat / 255.0)[..., None] * 0.65
    blended = (np.array(crop) * (1 - alpha) + colored * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def build_ui(engine: Engine) -> gr.Blocks:
    info = engine.info()
    queries = engine.config.get("example_queries") or DEFAULT_QUERIES
    crop_pct = engine.model.image_encoder.data_cfg.get("crop_pct", 0.875)
    has_captions = bool(engine.captions)

    def gallery_value(results):
        return [(r["path"], f"{r['score']:.3f}") for r in results]

    # ---------------- callbacks ----------------
    def text_search(query, k):
        if not query.strip():
            raise gr.Error("Type something to search for")
        res = engine.search_text(query, int(k))
        return gallery_value(res), res, query, gr.update(visible=False)

    def on_select(results, query, evt: gr.SelectData):
        r = results[evt.index]
        caps = "\n".join(f"- {c}" for c in r["captions"]) or "_No captions for this image_"
        md = f"**Similarity:** {r['score']:.3f}\n\n**Human-written captions:**\n{caps}"
        return gr.update(visible=True), r["path"], md, r["index"], None, None

    def explain(query, index):
        ex = engine.explain(query, int(index))
        words = [(w, "↑ important" if s > 0.02 else ("↓ hurts match" if s < -0.02 else None))
                 for w, s in ex["words"]]
        return overlay_heatmap(engine.image_path(int(index)), ex["heatmap"], crop_pct), words

    def image_search(image, k):
        if image is None:
            raise gr.Error("Upload an image first")
        res = engine.search_image(image, int(k))
        desc = engine.describe(image, 5)
        md = "\n".join(f"{i + 1}. {d['caption']} ({d['score']:.3f})" for i, d in enumerate(desc))
        return gallery_value(res), md or "_No caption bank in this gallery_"

    def classify(image, labels, template):
        if image is None:
            raise gr.Error("Upload an image first")
        if "{}" not in template:
            raise gr.Error("The template must contain {} where the label goes")
        names = [x for x in labels.split(",") if x.strip()]
        if len(names) < 2:
            raise gr.Error("Give at least two labels, separated by commas")
        return engine.classify(image, names, template)

    def map_frame():
        m = engine.map
        rows = []
        for j, (i, (x, y)) in enumerate(zip(m["index"], m["xy"])):
            cap = engine.items[i]["captions"][0] if engine.items[i]["captions"] else engine.items[i]["source"]
            rows.append({"x": x, "y": y, "theme": m["cluster_names"][m["cluster"][j]], "caption": cap})
        return pd.DataFrame(rows)

    def cluster_samples(name):
        m = engine.map
        c = m["cluster_names"].index(name)
        idx = [i for i, lab in enumerate(m["all_clusters"]) if lab == c]
        pick = np.random.default_rng().choice(idx, min(12, len(idx)), replace=False)
        return [(engine.image_path(int(i)), None) for i in pick]

    # ---------------- layout ----------------
    with gr.Blocks(title="Mini-CLIP · image search with contrastive learning") as demo:
        gr.Markdown(
            f"# Mini-CLIP image search\n"
            f"Search **{info['gallery_images']:,} photos** by describing them in plain English. "
            f"Images and text are mapped into one shared embedding space by a two-tower model trained with "
            f"contrastive learning ({info['image_backbone']} + {info['text_backbone'].split('/')[-1]}).",
            elem_id="hero")

        with gr.Tabs():
            # ---- Text search ----
            with gr.Tab("🔎 Search by text"):
                with gr.Row():
                    query = gr.Textbox(placeholder="Describe the photo you're looking for…", show_label=False,
                                       scale=5, value=queries[0])
                    go = gr.Button("Search", variant="primary", scale=1)
                with gr.Row():
                    gr.Examples(queries, inputs=query, label="Try")
                    k = gr.Slider(4, 40, value=16, step=4, label="Results")
                results_state, query_state, index_state = gr.State([]), gr.State(""), gr.State(0)
                gallery = gr.Gallery(label="Results (click an image for details)", columns=4, height="auto",
                                     object_fit="cover", allow_preview=False)
                with gr.Group(visible=False) as detail:
                    with gr.Row():
                        detail_img = gr.Image(label="Selected", height=320, interactive=False)
                        with gr.Column():
                            detail_md = gr.Markdown()
                            why = gr.Button("Why did this match? (explain)", variant="secondary")
                            gr.Markdown("Occlusion analysis: parts of the image and words of the query are "
                                        "hidden one at a time to measure how much each one matters.",
                                        elem_classes="small-note")
                    with gr.Row():
                        heat_img = gr.Image(label="Image regions that matter most", height=320,
                                            interactive=False)
                        words_out = gr.HighlightedText(label="Words that matter most",
                                                       color_map={"↑ important": "green",
                                                                  "↓ hurts match": "red"})
                for trigger in (go.click, query.submit):
                    trigger(text_search, [query, k], [gallery, results_state, query_state, detail])
                gallery.select(on_select, [results_state, query_state],
                               [detail, detail_img, detail_md, index_state, heat_img, words_out])
                why.click(explain, [query_state, index_state], [heat_img, words_out])

            # ---- Image search ----
            with gr.Tab("🖼️ Search by image"):
                gr.Markdown("Upload a photo (or use your webcam) to find similar photos in the collection"
                            + (", and the captions that describe it best." if has_captions else "."))
                with gr.Row():
                    with gr.Column(scale=2):
                        up = gr.Image(type="pil", label="Your image", sources=["upload", "webcam", "clipboard"],
                                      height=300)
                        k2 = gr.Slider(4, 40, value=12, step=4, label="Results")
                        go2 = gr.Button("Find similar", variant="primary")
                        desc_md = gr.Markdown(label="Best-matching captions")
                    with gr.Column(scale=3):
                        gallery2 = gr.Gallery(label="Similar photos", columns=3, height="auto",
                                              object_fit="cover")
                go2.click(image_search, [up, k2], [gallery2, desc_md])
                up.upload(image_search, [up, k2], [gallery2, desc_md])

            # ---- Zero-shot classifier ----
            with gr.Tab("🏷️ Zero-shot classifier"):
                gr.Markdown("Classify an image into **any categories you type**, with no extra training. Each "
                            "label becomes a sentence (e.g. *a photo of a dog*) and is compared with the image.")
                with gr.Row():
                    with gr.Column():
                        up3 = gr.Image(type="pil", label="Image", sources=["upload", "webcam", "clipboard"],
                                       height=300)
                        labels = gr.Textbox(label="Labels (comma-separated)", value=LABEL_SETS[0])
                        gr.Examples(LABEL_SETS, inputs=labels, label="Label sets")
                        template = gr.Textbox(label="Prompt template", value="a photo of {}")
                        go3 = gr.Button("Classify", variant="primary")
                    out3 = gr.Label(label="Prediction", num_top_classes=10)
                go3.click(classify, [up3, labels, template], out3)

            # ---- Embedding map ----
            if engine.map:
                with gr.Tab("🗺️ Embedding map"):
                    gr.Markdown("Every dot is a photo, placed by t-SNE so that photos the model considers "
                                "similar sit close together. Colours are clusters found by k-means and named "
                                "after their most common caption words. Hover to read captions.")
                    gr.ScatterPlot(map_frame(), x="x", y="y", color="theme", tooltip=["caption", "theme"],
                                   height=560, x_title=" ", y_title=" ", show_label=False)
                    with gr.Row():
                        theme = gr.Dropdown(engine.map["cluster_names"], label="Show photos from a cluster",
                                            value=engine.map["cluster_names"][0])
                        show = gr.Button("Show photos")
                    cluster_gallery = gr.Gallery(columns=6, height="auto", object_fit="cover", show_label=False)
                    show.click(cluster_samples, theme, cluster_gallery)
                    theme.change(cluster_samples, theme, cluster_gallery)

            # ---- About ----
            with gr.Tab("📄 About & API"):
                ev = info.get("eval") or []
                eval_md = ""
                if ev:
                    rows = "\n".join(
                        f"| {e['model']} | {e['metrics']['t2i_R@1']:.1f} | {e['metrics']['t2i_R@5']:.1f} | "
                        f"{e['metrics']['t2i_R@10']:.1f} | {e['metrics']['i2t_R@1']:.1f} | "
                        f"{e['metrics']['i2t_R@5']:.1f} |" for e in ev)
                    eval_md = ("### Test-set retrieval results (1,000 unseen images)\n"
                               "| Model | Text→Image R@1 | R@5 | R@10 | Image→Text R@1 | R@5 |\n"
                               "|---|---|---|---|---|---|\n" + rows)
                gr.Markdown(f"""
### How it works
Two encoders turn images and text into vectors: a pretrained **{info['image_backbone']}** for images and a
pretrained **{info['text_backbone']}** for text. Small **projection heads** map both into a shared
{info['embed_dim']}-dimensional space. They are trained with the **contrastive InfoNCE loss**: in each batch,
every photo must pick out its own caption among all captions in the batch, and vice versa, which pulls matching
pairs together and pushes others apart. Search is then a nearest-neighbour lookup by cosine similarity
({info['search_backend']} index).

| | |
|---|---|
| Training mode | {info['mode']} |
| Trained parameters | {info['trained_parameters'] / 1e6:.2f}M of {info['total_parameters'] / 1e6:.1f}M |
| Learned temperature | {info['temperature']} |
| Gallery | {info['gallery_images']:,} images, {info['caption_bank']:,} captions |

{eval_md}

### Limitations
The model was trained on a few thousand captioned photos, mostly of people and animals outdoors, so it works
best on similar scenes and can confuse fine details such as colours of specific objects or counts.

### API
Interactive documentation: [`/docs`](/docs)

```bash
curl -X POST BASE_URL/api/search/text -H "Content-Type: application/json" \\
     -d '{{"query": "a dog jumping into a lake", "k": 5}}'
curl -X POST BASE_URL/api/classify -F "file=@photo.jpg" -F "labels=dog, cat, horse"
```
""")
        demo.load(text_search, [query, k], [gallery, results_state, query_state, detail])
    return demo
