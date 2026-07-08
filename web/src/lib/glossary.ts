import glossaryData from "@/data/glossary.json";

export type GlossaryEntry = {
  term: string;
  explanation: string;
};

export type GlossarySection = {
  title: string;
  slug: string;
  entries: GlossaryEntry[];
};

export type GlossaryData = {
  intro: string;
  sections: GlossarySection[];
};

export function getGlossary(): GlossaryData {
  return glossaryData as GlossaryData;
}

export function getGlossaryStats(data: GlossaryData) {
  const entryCount = data.sections.reduce(
    (total, section) => total + section.entries.length,
    0,
  );

  return {
    entryCount,
    sectionCount: data.sections.length,
  };
}
