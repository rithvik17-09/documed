import { NextResponse } from "next/server";
import { prisma } from "../../../../lib/prisma";
import { sendSignupMail } from "../../../../lib/mailer";

export async function POST(req: Request) {
  const { username, password, gmail } = await req.json();
  if (!username || !password || !gmail) {
    return NextResponse.json({ error: "Username, password, and Gmail are required" }, { status: 400 });
  }
  // Check if username already exists
  const existingByUsername = await prisma.user.findUnique({ where: { username } });
  if (existingByUsername) {
    return NextResponse.json({ error: "User already exists" }, { status: 409 });
  }

  // Check if gmail already exists and return a clear message we can show in the UI
  const existingByGmail = await prisma.user.findUnique({ where: { gmail } });
  if (existingByGmail) {
    return NextResponse.json({ error: "Gmail already exists please login" }, { status: 409 });
  }

  await prisma.user.create({ data: { username, password, gmail } });
  try {
    await sendSignupMail(gmail, username);
  } catch (e) {
    // Optionally log error, but don't block signup
    console.error('Failed to send signup email:', e);
  }
  return NextResponse.json({ success: true });
}
