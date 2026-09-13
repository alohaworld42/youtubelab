# Varianten

Frühere/parallele Anläufe auf dasselbe Ziel, als eigene Branches gesichert.
Nicht in den Hauptbranch gemergt.

| Branch | Ordner lokal | Was es war | Stack | Warum verworfen |
|---|---|---|---|---|
| `variante/youtubevideogenerator` | `Desktop/projects/youtubevideogenerator` | Erster Anlauf (Feb 2026): faceless YouTube Shorts über den Open-Source-Server `short-video-maker` + n8n, orchestriert von einem einzelnen `generate_video.js` (Reddit/Gemini -> Szenen -> Kokoro-TTS -> Remotion-Render) | Node.js (dependency-frei), Docker Compose, n8n, Gemini 2.5 Flash, Pexels | Abhängig von fremdem Docker-Server und n8n; keine eigene Kontrolle über Rendering, Captions, Upload, kein Multi-Channel und keine Review-Queue. Abgelöst durch die eigene Python/Flask-Pipeline in diesem Repo. |

Nicht mitgesichert im Varianten-Branch (Fremdcode bzw. Grossdaten, lokal weiterhin vorhanden):
`short-video-maker/` und `ai_agents_az/` (eigenständige Upstream-git-Clones), `videos/`, `n8n_data/`, `*.mp4`, `.env`.
