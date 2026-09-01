# YouTube Upload Setup

VideoGen can push finished videos straight to your YouTube channel (one click on the job page, uploaded **private** by default so you review in YouTube Studio before publishing). This requires a one-time Google Cloud setup — about 10 minutes.

The app requests only the `https://www.googleapis.com/auth/youtube.upload` scope: it can upload videos and set their thumbnails, and nothing else (it cannot read your channel, comments, or analytics).

## 1. Create a Google Cloud project

1. Open the [Google Cloud Console](https://console.cloud.google.com/) signed in with the Google account that owns your YouTube channel.
2. Project picker (top bar) → **New project** → name it (e.g. `YouTube Uploader`) → **Create**, then select it.

## 2. Enable the YouTube Data API

1. **APIs & Services → Library**.
2. Search **YouTube Data API v3** → **Enable**.

## 3. Configure the OAuth consent screen

1. **APIs & Services → OAuth consent screen** (newer consoles: **Google Auth Platform → Branding/Audience**).
2. User type: **External** → fill in the app name and your email → save through the steps (no scopes need to be added here; the app requests its scope at sign-in).
3. **Important — pick one:**
   - **Testing mode (quickest):** under **Test users**, click **+ Add users** and add the Google account email you will sign in with. Without this you get `Error 403: access_denied` at sign-in. Note: in Testing mode Google **expires the stored token after 7 days**, so you'll need to click *Connect YouTube* again about once a week.
   - **Publish the app (set-and-forget):** click **Publish app** to move it to *In production*. For a personal tool you do **not** need to complete Google's full verification — the sign-in shows a "Google hasn't verified this app" warning that you pass via **Advanced → Go to <app> (unsafe)** — and in return the token does not expire.

## 4. Create the OAuth client

1. **APIs & Services → Credentials → + Create credentials → OAuth client ID**.
2. Application type: **Desktop app** (this matters — Desktop clients allow the localhost redirect VideoGen uses). Name it anything.
3. **Create**, then **Download JSON**.
4. Save the downloaded file as **`client_secret.json`** in the VideoGen project folder (next to `app.py`).

Both `client_secret.json` and the stored token (`youtube_token.json`) are gitignored — they never leave your machine or enter the repository.

## 5. Connect VideoGen

1. Open VideoGen → **Settings** → the **YouTube** card should now say *not connected*.
2. Click **Connect YouTube** → a Google sign-in tab opens → choose the channel-owning account → allow the upload permission → the tab confirms "YouTube connected".
3. The Settings card switches to *connected*. Done — this persists across restarts.

## 6. Upload a video

On any **finished** job page, click **Upload to YouTube**. The upload runs in the background with live progress; when it completes, the job shows the `youtu.be/...` link. The video arrives with:

- your configured **privacy** (Settings → *YouTube privacy*; default **private** — an unpublished draft you review and publish from YouTube Studio),
- the **made for kids** flag (Settings toggle; on by default — the COPPA self-declaration),
- the **SEO title and description** generated for the job,
- the generated **custom thumbnail** (requires a phone-verified channel; if not verified, the video still uploads and a note is logged).

Re-uploading is allowed (e.g. after regenerating clips) — the app warns you explicitly that a second copy will be created.

## Troubleshooting

- **`Error 403: access_denied` at sign-in** — your account is not a test user (Testing mode). Add it under *OAuth consent screen → Test users*, or publish the app (step 3).
- **"Google hasn't verified this app"** — expected for an unverified personal project; click *Advanced → Go to … (unsafe)* to continue.
- **Uploads stop working after ~a week** — Testing-mode token expiry; reconnect in Settings, or publish the app to stop the expiry.
- **`redirect_uri_mismatch`** — the OAuth client is not a *Desktop app* type; create a new client with the correct type and replace `client_secret.json`.
- **Thumbnail not set, video uploaded** — the channel isn't verified for custom thumbnails; verify it at [youtube.com/verify](https://www.youtube.com/verify).
- **Quota**: the YouTube API grants 10,000 units/day by default; one upload costs ~1,600 — roughly 6 uploads/day. Request more in the Cloud Console if needed.
