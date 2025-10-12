
import { getServerSession } from "next-auth/next";
import { authOptions } from "../app/api/auth/[...nextauth]/route";
import { redirect } from "next/navigation";


export default async function AuthGuard({ children }: { children: React.ReactNode }) {
  const session = await getServerSession(authOptions);
  if (!session) {
    redirect("/auth");
  }
  return <>{children}</>;
}
