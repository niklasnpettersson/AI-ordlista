import { GlossaryApp } from "@/components/glossary-app";
import { getGlossary, getGlossaryStats } from "@/lib/glossary";

export default function HomePage() {
  const glossary = getGlossary();
  const stats = getGlossaryStats(glossary);

  return (
    <GlossaryApp
      data={glossary}
      entryCount={stats.entryCount}
      sectionCount={stats.sectionCount}
    />
  );
}
