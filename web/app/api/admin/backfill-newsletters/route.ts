export const runtime = 'nodejs';

export async function POST(req: Request) {
  const { searchParams } = new URL(req.url);

  const serverBase = process.env.PY_SERVER_URL || 'http://localhost:8001';
  const upstream = new URL(`${serverBase.replace(/\/$/, '')}/api/v1/admin/backfill-newsletters`);

  // Forward all query params (gmail_query, max_results, force) to FastAPI
  searchParams.forEach((value, key) => upstream.searchParams.set(key, value));

  try {
    const resp = await fetch(upstream.toString(), { method: 'POST' });
    const data = await resp.json().catch(() => ({}));
    return new Response(JSON.stringify(data), {
      status: resp.status,
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
    });
  } catch (e: any) {
    return new Response(
      JSON.stringify({ ok: false, error: 'Upstream error', detail: e?.message || String(e) }),
      { status: 502, headers: { 'Content-Type': 'application/json; charset=utf-8' } }
    );
  }
}
