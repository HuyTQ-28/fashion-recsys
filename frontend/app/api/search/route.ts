import { NextResponse } from 'next/server';

const BACKEND_API_URL = process.env.BACKEND_API_URL ?? 'http://localhost:8000';

export async function POST(req: Request) {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  try {
    const payload = await req.json();
    const controller = new AbortController();
    timeout = setTimeout(() => controller.abort(), 12000);

    const response = await fetch(`${BACKEND_API_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      cache: 'no-store',
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(
        { detail: data?.detail ?? 'Search request failed.' },
        { status: response.status },
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        { detail: 'Search backend timed out. Please retry.' },
        { status: 504 },
      );
    }
    return NextResponse.json(
      { detail: 'Unable to reach backend search service.' },
      { status: 502 },
    );
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}
