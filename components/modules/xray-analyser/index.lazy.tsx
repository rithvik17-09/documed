import dynamic from "next/dynamic";

const XrayAnalyser = dynamic(() => import("@/components/modules/xray-analyser"), { ssr: false });

export default XrayAnalyser;
