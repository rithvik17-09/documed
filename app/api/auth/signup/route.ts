import { NextResponse } from "next/server";
import { prisma } from "../../../../lib/prisma";
import { sendSignupMail } from "../../../../lib/mailer";

export async function POST(req: Request) {
  const { username, password, gmail } = await req.json();
  if (!username || !password || !gmail) {
    return NextResponse.json({ error: "Username, password, and Gmail are required" }, { status: 400 });
  }
  const existing = await prisma.user.findUnique({ where: { username } });
  if (existing) {
    return NextResponse.json({ error: "User already exists" }, { status: 409 });
  }
  await prisma.user.create({ data: { username, password, gmail } });
  try {
    await sendSignupMail(gmail);
  } catch (e) {
    // Optionally log error, but don't block signup
    console.error('Failed to send signup email:', e);
  }
  return NextResponse.json({ success: true });
}
