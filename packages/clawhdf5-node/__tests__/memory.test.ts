/**
 * Integration tests for @redclaw/clawhdf5.
 *
 * These tests require the native addon to be built first:
 *   npm run build
 *
 * They write temporary .h5 files to the OS temp directory and clean up after
 * themselves.
 */

import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import { ClawhdfMemory, MemorySearchResult, BackendStats, ConsolidationStats } from '../src/index';

// Helper: create a unique temp path for each test.
function tmpBrain(name: string): string {
  return path.join(os.tmpdir(), `clawhdf5-test-${name}-${Date.now()}.brain`);
}

// Helper: clean up after test.
function cleanup(...paths: string[]): void {
  for (const p of paths) {
    for (const ext of ['', '.wal']) {
      try { fs.unlinkSync(p + ext); } catch { /* ignore */ }
    }
  }
}

describe('ClawhdfMemory — lifecycle', () => {
  test('create() produces an empty store', () => {
    const p = tmpBrain('create');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      const s = mem.stats();
      expect(s.totalRecords).toBe(0);
      expect(s.totalEmbeddings).toBe(0);
      expect(s.modalities).toEqual([]);
      expect(s.lastUpdated).toBeUndefined();
    } finally {
      cleanup(p);
    }
  });

  test('openOrCreate() creates when absent', () => {
    const p = tmpBrain('openOrCreate-new');
    try {
      expect(fs.existsSync(p)).toBe(false);
      const mem = ClawhdfMemory.openOrCreate(p, 4);
      expect(fs.existsSync(p)).toBe(true);
      expect(mem.stats().totalRecords).toBe(0);
    } finally {
      cleanup(p);
    }
  });

  test('open() reloads persisted data', () => {
    const p = tmpBrain('open-persist');
    try {
      {
        const mem = ClawhdfMemory.create(p, 4);
        mem.write('note.md', 'persisted content');
      }
      const mem2 = ClawhdfMemory.open(p);
      const result = mem2.get('note.md');
      expect(result).not.toBeNull();
      expect(result).toContain('persisted content');
    } finally {
      cleanup(p);
    }
  });

  test('openOrCreate() opens when present', () => {
    const p = tmpBrain('openOrCreate-exists');
    try {
      {
        const mem = ClawhdfMemory.create(p, 4);
        mem.write('x.md', 'some data');
      }
      const mem2 = ClawhdfMemory.openOrCreate(p, 4);
      expect(mem2.stats().totalRecords).toBe(1);
    } finally {
      cleanup(p);
    }
  });
});

describe('ClawhdfMemory — write / get', () => {
  test('write then get returns the content', () => {
    const p = tmpBrain('write-get');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      mem.write('memory/notes.md', 'Hello, memory!');
      const result = mem.get('memory/notes.md');
      expect(result).toContain('Hello, memory!');
    } finally {
      cleanup(p);
    }
  });

  test('get returns null for unknown path', () => {
    const p = tmpBrain('get-missing');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      expect(mem.get('does/not/exist.md')).toBeNull();
    } finally {
      cleanup(p);
    }
  });

  test('get with line range filters correctly', () => {
    const p = tmpBrain('get-linerange');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      mem.write('lines.md', 'line0\nline1\nline2\nline3\nline4');
      const slice = mem.get('lines.md', 1, 2);
      expect(slice).not.toBeNull();
      const lines = slice!.split('\n');
      expect(lines).toHaveLength(2);
      expect(lines[0]).toBe('line1');
      expect(lines[1]).toBe('line2');
    } finally {
      cleanup(p);
    }
  });
});

describe('ClawhdfMemory — stats', () => {
  test('stats update after writes', () => {
    const p = tmpBrain('stats');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      mem.write('a.md', 'content A');
      mem.write('b.md', 'content B');
      const s = mem.stats();
      expect(s.totalRecords).toBe(2);
      expect(s.modalities).toContain('text');
      expect(s.fileSizeBytes).toBeGreaterThan(0);
      expect(s.lastUpdated).toBeDefined();
    } finally {
      cleanup(p);
    }
  });
});

describe('ClawhdfMemory — Markdown ingestion & export', () => {
  test('ingestMarkdown returns section count', () => {
    const p = tmpBrain('ingest');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      const md = '# Goals\n\nLearn Rust.\n\n# Projects\n\nBuild clawhdf5.';
      const count = mem.ingestMarkdown('memory/user.md', md);
      expect(count).toBe(2);
      expect(mem.stats().totalRecords).toBe(2);
    } finally {
      cleanup(p);
    }
  });

  test('ingestMarkdown empty string returns 0', () => {
    const p = tmpBrain('ingest-empty');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      expect(mem.ingestMarkdown('empty.md', '')).toBe(0);
    } finally {
      cleanup(p);
    }
  });

  test('exportMarkdown roundtrip preserves sections', () => {
    const p = tmpBrain('export');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      const md = '## Section One\n\nContent one.\n\n## Section Two\n\nContent two.';
      mem.ingestMarkdown('docs/test.md', md);
      const exported = mem.exportMarkdown('docs/test.md');
      expect(exported).toContain('Section One');
      expect(exported).toContain('Content one.');
      expect(exported).toContain('Section Two');
      expect(exported).toContain('Content two.');
    } finally {
      cleanup(p);
    }
  });

  test('exportMarkdown throws for unknown path', () => {
    const p = tmpBrain('export-missing');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      expect(() => mem.exportMarkdown('missing.md')).toThrow();
    } finally {
      cleanup(p);
    }
  });
});

describe('ClawhdfMemory — search', () => {
  test('search on empty store returns empty array', () => {
    const p = tmpBrain('search-empty');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      const results = mem.search('anything', new Float32Array(4), 5);
      expect(results).toEqual([]);
    } finally {
      cleanup(p);
    }
  });

  test('search result fields are populated', () => {
    const p = tmpBrain('search-fields');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      mem.write('mem/x.md', 'hello world greeting');
      const results = mem.search('hello', new Float32Array(4), 3);
      if (results.length > 0) {
        const r = results[0];
        expect(typeof r.text).toBe('string');
        expect(r.text.length).toBeGreaterThan(0);
        expect(typeof r.score).toBe('number');
        expect(typeof r.path).toBe('string');
        expect(typeof r.source).toBe('string');
      }
    } finally {
      cleanup(p);
    }
  });
});

describe('ClawhdfMemory — consolidation hooks', () => {
  test('compact() returns a number', () => {
    const p = tmpBrain('compact');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      mem.write('x.md', 'some content');
      const removed = mem.compact();
      expect(typeof removed).toBe('number');
      expect(removed).toBeGreaterThanOrEqual(0);
    } finally {
      cleanup(p);
    }
  });

  test('tickSession() does not throw', () => {
    const p = tmpBrain('tick');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      mem.write('y.md', 'tick test');
      expect(() => mem.tickSession()).not.toThrow();
    } finally {
      cleanup(p);
    }
  });

  test('flushWal() does not throw', () => {
    const p = tmpBrain('flushwal');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      expect(() => mem.flushWal()).not.toThrow();
    } finally {
      cleanup(p);
    }
  });

  test('runConsolidation() returns correct shape', () => {
    const p = tmpBrain('consolidate');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      mem.write('a.md', 'alpha content');
      mem.write('b.md', 'beta content');
      const stats: ConsolidationStats = mem.runConsolidation(Date.now() / 1000);
      expect(typeof stats.workingCount).toBe('number');
      expect(typeof stats.episodicCount).toBe('number');
      expect(typeof stats.semanticCount).toBe('number');
      expect(typeof stats.totalEvictions).toBe('number');
      expect(typeof stats.totalPromotions).toBe('number');
    } finally {
      cleanup(p);
    }
  });

  test('walPendingCount() returns a non-negative number', () => {
    const p = tmpBrain('walpending');
    try {
      const mem = ClawhdfMemory.create(p, 4);
      expect(mem.walPendingCount()).toBeGreaterThanOrEqual(0);
    } finally {
      cleanup(p);
    }
  });
});
