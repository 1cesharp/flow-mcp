# Parking lot

Ideas we want, not started yet. Newest at the bottom.

## Turn landscape videos into vertical reels and shorts

Take a wide 16:9 film (from Cut, or any video) and make a 9:16 version for Instagram Reels, YouTube Shorts
and TikTok, without regenerating anything in Flow.

- **Ways to fill the tall frame**, the user picks one:
  - *Follow the subject*: crop to whoever or whatever matters and keep them centred as they move. The best
    result, and the hardest part: it needs something that finds the person or subject in each shot.
  - *Centre crop*: the middle of the frame, full height. Simple, but anything near the edges is lost.
  - *Fit with a blurred background*: the whole wide picture in the middle, with a blurred, enlarged copy
    filling the space above and below. Nothing is cut off, and the empty bands suit captions.
- **Captions go with it**: the vertical version should offer captions in the same step, placed where they
  read best for the layout chosen (in the blurred band, for example).
- **Preview first**: show a frame of the vertical result before making it, like the Captions tab does.
- **Where it lives**: a Studio tab (or an option on Cut's finished film) and a matching MCP tool, so Claude
  can do it too.
- **How it could be built**: FFmpeg already does centre crop and blurred fill with no new installs. Following
  the subject needs detection added on top, done locally like the captions so nothing is uploaded.
- **Worth noting**: Flow can also make clips vertical from the start (9:16 in Shots), which beats cropping
  when the vertical version is planned up front. This idea is for films that were made wide.
