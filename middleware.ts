import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { getToken } from "next-auth/jwt";

export default async function middleware(request: NextRequest) {
  const token = await getToken({ req: request, secret: process.env.NEXTAUTH_SECRET });
  const isAuthPage = request.nextUrl.pathname.startsWith("/auth");

  if (!token && !isAuthPage) {
    const authUrl = new URL("/auth", request.url);
    return NextResponse.redirect(authUrl);
  }
  if (token && isAuthPage) {
    // Redirect authenticated users to the external main page
    return NextResponse.redirect("https://documed-o2av46fnk-rithviks-projects-14bc107b.vercel.app/");
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next|api|static|favicon.ico).*)"],
};
