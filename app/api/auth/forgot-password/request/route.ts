import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { sendOtpMail } from "@/lib/mailer";

// Request password reset (send OTP)
export async function PUT(req: Request) {
  const { gmail } = await req.json();
  const user = await prisma.user.findUnique({ where: { gmail } });
  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }
  // Generate 6-digit OTP
  const otp = Math.floor(100000 + Math.random() * 900000).toString();
  const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes from now
  await prisma.passwordResetToken.create({ data: { otp, userId: user.id, expiresAt } });
  try {
  await sendOtpMail(gmail, otp);
  } catch (e) {
    console.error('Failed to send OTP email:', e);
    return NextResponse.json({ error: "Failed to send OTP email" }, { status: 500 });
  }
  return NextResponse.json({ success: true });
}
