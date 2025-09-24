
import { NextResponse } from "next/server";

export function PUT() {
	return NextResponse.json({ error: "Use /api/auth/forgot-password/request for password reset requests." }, { status: 405 });
}

export function PATCH() {
	return NextResponse.json({ error: "Use /api/auth/forgot-password/reset for password reset confirmation." }, { status: 405 });
}
