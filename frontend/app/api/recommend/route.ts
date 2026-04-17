import { NextResponse } from 'next/server';

const BACKEND_API_URL = process.env.BACKEND_API_URL ?? 'http://localhost:8000';

export async function POST(req: Request) {
  try {
    const payload = await req.json();
    const response = await fetch(`${BACKEND_API_URL}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      cache: 'no-store',
    });

    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(
        { detail: data?.detail ?? 'Recommendation request failed.' },
        { status: response.status },
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch {
    return NextResponse.json(
      { detail: 'Unable to reach backend recommendation service.' },
      { status: 502 },
    );
  }
}
