// /api/openai-ping.ts
export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return res.status(405).json({ ok: false, error: "Method not allowed" });
  const { apiKey } = (req.body || {});
  if (!apiKey) return res.status(400).json({ ok: false, error: "Missing apiKey" });

  try {
    const r = await fetch("https://api.openai.com/v1/models?limit=1", {
      headers: { Authorization: `Bearer ${apiKey}` },
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) return res.status(r.status).json({ ok: false, error: data?.error?.message || "Failed" });
    return res.status(200).json({ ok: true, sampleModel: data?.data?.[0]?.id || "ok" });
  } catch (e: any) {
    return res.status(500).json({ ok: false, error: e?.message || "Network error" });
  }
}
