
import { getServerSession } from "next-auth/next";
import { authOptions } from "../app/api/auth/[...nextauth]/route";
import { redirect, usePathname } from "next/navigation";


export default async function AuthGuard({ children }: { children: React.ReactNode }) {
  // Only protect if not on /auth/signin
  const pathname = typeof window !== 'undefined' ? window.location.pathname : '';
  if (pathname === "/auth/signin") {
    return <>{children}</>;
  }
  const session = await getServerSession(authOptions);
  if (!session) {
    redirect("/auth/signin");
  }
  return <>{children}</>;
}
