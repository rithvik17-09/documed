import type { NextApiRequest, NextApiResponse } from 'next';

const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent";

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { query } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'Missing query' });
  }

  const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY || process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'Missing Gemini API key' });
  }

  try {
    const geminiRes = await fetch(`${GEMINI_API_URL}?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: query }] }],
      }),
    });
    const data = await geminiRes.json();
    res.status(200).json({
      gemini: data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || null,
      raw: data,
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to contact Gemini API' });
  }
}
