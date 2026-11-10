# SpecLane: illustration prompt

One prompt for one results poster, in the format of Pramana's infographic (title, subtitle, white cards with a
table and charts, footer notes) with a different theme: cream paper, deep teal and amber. Portrait 2:3, high
resolution. Every number comes from the report ([index.html](index.html)).

```
A polished research-report infographic poster, portrait 2:3, flat vector style. Background:
cream paper (#faf6ee) fading to pale teal (#e3f1ee) at the bottom, with a few soft, flowing
parallel "lane" lines in light teal across the lower area and two thin vertical pin markers
with small amber dots at the top right. Content sits on rounded white cards with 1 px light-grey
borders and very soft shadows. Inter-style sans-serif type, monospace for formulas. Generous
white space, strict grid alignment, 24 px gutters. Palette only: cream #faf6ee, white, ink
#10201f, grey #5b6b69, deep teal #0f766e, light teal #99d5cf, pale teal #e3f1ee, amber #e0a030,
light grey #c9c6bd, red #c2410c (FP8 series only).

HEADER (left): bold two-line title "SpecLane: Faster Serving for Small Language Models" with a
short amber underline bar. Grey subtitle: "Multi-token prediction (MTP) on Qwen3.8-27B and
Qwen3.5-9B, served with vLLM". On the right, three stat tiles: "+124%" / "27B, one request at a
time"; "+77%" / "9B, one request at a time"; "+/-2 of 200" / "change in answer accuracy".

CARD 1, header "Throughput on agent traffic", sub-line "Output tokens per second, higher is
better; change from baseline in brackets". Table, columns Model | Setting | 1 request | 16
requests | p99 first token (s):
  Qwen3.8-27B | Baseline | 78 | 800 | 1.09
  Qwen3.8-27B | MTP, 3 proposed tokens | 175 (+124%) | 1,199 (+50%) | 0.94
  Qwen3.8-27B | FP8 KV cache | 76 (-2%) | 684 (-14%) | 1.46
  Qwen3.5-9B | Baseline | 43 | 475 | 0.95
  Qwen3.5-9B | MTP, 2 proposed tokens | 75 (+77%) | 753 (+59%) | 0.74
  Qwen3.5-9B | FP8 KV cache | 43 (+0%) | 467 (-2%) | 1.42
The two MTP rows have a pale teal background (#e3f1ee) and bold deep-teal numbers.

CARD 2 (left), header "Throughput by workload, 16 requests at once", sub-line "Output tokens
per second". Grouped bar chart, y-axis 0 to 1,400, four groups of three bars (Baseline light
grey, MTP deep teal, FP8 KV cache amber): "27B, shared prompt" 743, 1,297, 612; "27B, agent
sessions" 800, 1,199, 684; "9B, shared prompt" 420, 644, 399; "9B, agent sessions" 475, 753,
467. Value labels on the MTP bars only; legend at the top.

CARD 3 (right), header "First-token delay, 16 requests at once", sub-line "p99 time to first
token in seconds, lower is better". Grouped bar chart, y-axis 0 to 1.6, same colours and groups:
"27B, shared prompt" 0.53, 0.79, 1.47; "27B, agent sessions" 1.09, 0.94, 1.46; "9B, shared
prompt" 0.69, 0.84, 1.39; "9B, agent sessions" 0.95, 0.74, 1.42. Value labels on all bars.

CARD 4 (left), header "How the speed-up works", sub-line "A small head proposes, the full model
checks". A three-step flow of rounded boxes with arrows: "MTP head proposes k tokens" (three amber
chips) -> "Full model checks all k in one pass" (two teal ticks, one red cross) -> "Keep the
accepted tokens, plus one more; output unchanged". Beneath it, on a pale teal panel in
monospace: "Tokens per step = (1 - a^(k+1)) / (1 - a)" and "Speed-up = tokens per step / (1 +
c*k)", with the caption "a = chance a proposal is accepted, c = proposal cost vs one full-model
step".

CARD 5 (right), header "Why there is a best k", sub-line "Illustrative: a = 0.6, c = 0.1, not a
measurement". Line chart with markers, x = k from 1 to 6, y-axis 1.0 to 1.8, points 1.46, 1.63,
1.67, 1.65, 1.59, 1.52; the k = 3 point enlarged, deep teal, labelled "best: k = 3".

CARD 6 (full width, slim), header "Answer quality is preserved", sub-line "200 GSM8K questions,
exact match". Table, columns Model | Baseline | MTP | Difference: "Qwen3.8-27B-FP8" | 0.940 |
0.935 | -1 question; "Qwen3.5-9B" | 0.900 | 0.910 | +2 questions.

FOOTER, small grey text with an info icon: "One run per measurement: 24 requests (96 for 16
concurrent conversations). One NVIDIA H100 (27B) and one L40S (9B), vLLM 0.33.0. MTP =
multi-token prediction, built into each model. KV cache = stored attention keys and values."
Bottom right, bold small: "SpecLane".

Chart rules: bar heights and line points exactly proportional to the numbers on zero-based
axes (except the speed-up chart axis stated above), light gridlines, no 3-D, no gradients on
bars. Do not add logos, photos, people, watermarks, extra icons, or any text or number not given
above. Do not round or reorder anything.
```

Tips: image models can change digits, so compare the result with the report before using it. If a number comes out
wrong, regenerate with that card only, or fix the text in a vector editor. To copy Pramana's look closely, attach its
image as a layout reference and add "use the same card layout and spacing, with the theme and content described here."
