// SEMENTARA: uji coba e5-small di dalam Next.js. JANGAN di-commit, hapus setelah uji coba.
import os from "node:os";
import path from "node:path";
import { NextResponse } from "next/server";
import { pipeline, env } from "@huggingface/transformers";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Di Vercel hanya folder sementara (/tmp) yang boleh ditulisi, jadi model disimpan di sana.
env.cacheDir = path.join(os.tmpdir(), "edupuan-model");

const muat = () => pipeline("feature-extraction", "Xenova/multilingual-e5-small", { dtype: "q8" });
let model: ReturnType<typeof muat> | null = null;

export async function GET() {
  try {
    const pertamaKali = model === null;
    const t0 = Date.now();
    model ??= muat();
    const ekstrak = await model;
    const msMuat = Date.now() - t0;

    const t1 = Date.now();
    const h = await ekstrak("query: apakah penagih boleh menghubungi keluarga saya?", {
      pooling: "mean",
      normalize: true,
    });
    const v = (h.tolist() as number[][])[0];

    return NextResponse.json({ pertamaKali, dimensi: v.length, msMuat, msEmbed: Date.now() - t1 });
  } catch (e) {
    model = null; // supaya bisa dicoba lagi
    return NextResponse.json({ ok: false, detail: String(e) }, { status: 500 });
  }
}