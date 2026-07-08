"use client";

import { useMemo, useState } from "react";

import type { GlossaryData } from "@/lib/glossary";

type GlossaryAppProps = {
  data: GlossaryData;
  entryCount: number;
  sectionCount: number;
};

function normalize(value: string) {
  return value.toLowerCase().trim();
}

export function GlossaryApp({
  data,
  entryCount,
  sectionCount,
}: GlossaryAppProps) {
  const [query, setQuery] = useState("");
  const normalizedQuery = normalize(query);

  const visibleSections = useMemo(() => {
    if (!normalizedQuery) {
      return data.sections;
    }

    return data.sections
      .map((section) => ({
        ...section,
        entries: section.entries.filter((entry) => {
          const haystack = normalize(`${entry.term} ${entry.explanation}`);
          return haystack.includes(normalizedQuery);
        }),
      }))
      .filter((section) => section.entries.length > 0);
  }, [data.sections, normalizedQuery]);

  const visibleEntryCount = visibleSections.reduce(
    (total, section) => total + section.entries.length,
    0,
  );

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col lg:flex-row">
        <aside className="border-b border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900 lg:sticky lg:top-0 lg:h-screen lg:w-80 lg:overflow-y-auto lg:border-b-0 lg:border-r">
          <div className="space-y-4">
            <div>
              <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                AI-ordlista
              </p>
              <h1 className="text-2xl font-semibold tracking-tight">
                Svenska AI-begrepp
              </h1>
              <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
                {entryCount} termer · {sectionCount} sektioner
              </p>
            </div>

            <label className="block">
              <span className="sr-only">Sök i ordlistan</span>
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Sök term eller text…"
                className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none ring-blue-500/30 focus:border-blue-500 focus:ring-4 dark:border-slate-700 dark:bg-slate-950"
              />
            </label>

            <nav className="hidden max-h-[50vh] space-y-1 overflow-y-auto lg:block">
              {data.sections.map((section) => (
                <a
                  key={section.slug}
                  href={`#${section.slug}`}
                  className="block rounded-lg px-3 py-2 text-sm text-slate-700 transition hover:bg-blue-50 hover:text-blue-700 dark:text-slate-300 dark:hover:bg-blue-950 dark:hover:text-blue-300"
                >
                  {section.title}
                </a>
              ))}
            </nav>
          </div>
        </aside>

        <main className="flex-1 p-5 lg:p-8">
          <section className="mb-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            <p className="text-base leading-7 text-slate-600 dark:text-slate-300">
              {data.intro}
            </p>
          </section>

          {visibleSections.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-300 p-8 text-center text-slate-500 dark:border-slate-700 dark:text-slate-400">
              Inga träffar. Prova ett annat sökord.
            </div>
          ) : (
            <div className="space-y-10">
              {normalizedQuery ? (
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Visar {visibleEntryCount} träffar
                </p>
              ) : null}

              {visibleSections.map((section) => (
                <section
                  key={section.slug}
                  id={section.slug}
                  className="scroll-mt-6"
                >
                  <div className="mb-4 flex items-end justify-between gap-4">
                    <div>
                      <h2 className="text-2xl font-semibold tracking-tight">
                        {section.title}
                      </h2>
                      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                        {section.entries.length} termer
                      </p>
                    </div>
                  </div>

                  <div className="grid gap-4">
                    {section.entries.map((entry) => (
                      <article
                        key={`${section.slug}-${entry.term}`}
                        className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:border-blue-200 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-blue-900"
                      >
                        <h3 className="text-base font-semibold text-blue-700 dark:text-blue-300">
                          {entry.term}
                        </h3>
                        <p className="mt-2 text-sm leading-7 text-slate-700 dark:text-slate-300">
                          {entry.explanation}
                        </p>
                      </article>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
