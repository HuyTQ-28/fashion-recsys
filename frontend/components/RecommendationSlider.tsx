'use client';

import Image from 'next/image';
import { useRef, useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Sparkles } from 'lucide-react';
import { Product } from '@/components/ProductCard';

// ── Compact card inside the slider ─────────────────────────────
interface SliderCardProps {
  product: Product;
  index: number;
  onInteract: (id: string, type: 'click' | 'favorite' | 'purchase') => void;
}

function SliderCard({ product, index, onInteract }: SliderCardProps) {
  const mockPrice = (parseInt(product.article_id.slice(-4)) % 150) + 49;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.05, ease: 'easeOut' }}
      className="group flex-shrink-0 w-44 sm:w-52 cursor-pointer"
      onClick={() => onInteract(product.article_id, 'click')}
    >
      <div className="relative aspect-[3/4] rounded-xl overflow-hidden bg-gray-100 mb-3 transition-all duration-300 group-hover:shadow-md">
        {product.image_url ? (
          <Image
            src={product.image_url}
            alt={product.prod_name}
            fill
            className="object-cover object-top transition-all duration-500 ease-out group-hover:scale-105"
            sizes="208px"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-gray-400 uppercase tracking-widest">
            No Image
          </div>
        )}

        {/* AI score badge */}
        {product.score !== undefined && (
          <div className="absolute top-2 left-2 flex items-center gap-1 px-2 py-1 bg-white/95 backdrop-blur-sm rounded-full text-[10px] font-semibold text-emerald-600 shadow-sm">
            <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
            </svg>
            {Math.round(product.score * 100)}%
          </div>
        )}

        {/* "Personalized for you" tag */}
        {product.score !== undefined && product.score > 0.7 && (
          <div className="absolute bottom-2 left-2 right-2 py-1.5 px-2 bg-gradient-to-r from-violet-600 to-indigo-600 text-white text-[9px] font-bold tracking-widest uppercase rounded-lg text-center">
            Personalized for you
          </div>
        )}
      </div>

      <h4 className="text-sm font-medium text-gray-900 line-clamp-1 tracking-tight mb-0.5 group-hover:text-black transition-colors">
        {product.prod_name}
      </h4>
      <p className="text-xs text-gray-400 tracking-wide">
        {product.product_group_name || 'Fashion'} · <span className="font-medium text-gray-700">${mockPrice}</span>
      </p>
    </motion.div>
  );
}

// ── Main Slider component ───────────────────────────────────────
interface RecommendationSliderProps {
  products: Product[];
  loading: boolean;
  seedProduct: Product | null;
  /** How many interactions have been buffered but not yet applied to recommendations */
  pendingInteractions: number;
  onInteract: (id: string, type: 'click' | 'favorite' | 'purchase') => void;
}

export function RecommendationSlider({
  products,
  loading,
  seedProduct,
  pendingInteractions,
  onInteract,
}: RecommendationSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [canLeft, setCanLeft] = useState(false);
  const [canRight, setCanRight] = useState(true);

  const syncArrows = useCallback(() => {
    const el = trackRef.current;
    if (!el) return;
    setCanLeft(el.scrollLeft > 4);
    setCanRight(el.scrollLeft < el.scrollWidth - el.clientWidth - 4);
  }, []);

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    el.addEventListener('scroll', syncArrows, { passive: true });
    // Re-sync after products load
    syncArrows();
    return () => el.removeEventListener('scroll', syncArrows);
  }, [syncArrows, products]);

  const scroll = (dir: 'left' | 'right') => {
    trackRef.current?.scrollBy({ left: dir === 'left' ? -340 : 340, behavior: 'smooth' });
  };

  if (!loading && products.length === 0) return null;

  return (
    <motion.section
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mt-20 pt-14 border-t border-gray-100"
    >
      {/* ── Section header ───────────────────────────── */}
      <div className="flex items-start justify-between gap-4 mb-8">
        <div className="flex flex-col gap-2 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <Sparkles size={14} strokeWidth={1.5} className="text-black shrink-0" />
            <span className="text-[10px] font-semibold tracking-widest uppercase text-gray-400">
              AI Curated
            </span>

            {/* "Pending refresh" pill */}
            <AnimatePresence>
              {pendingInteractions > 0 && (
                <motion.span
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className="inline-flex items-center gap-1.5 px-2.5 py-0.5 bg-amber-50 border border-amber-200 text-amber-600 text-[9px] font-semibold tracking-widest uppercase rounded-full"
                >
                  <motion.span
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ repeat: Infinity, duration: 1.2 }}
                    className="w-1 h-1 rounded-full bg-amber-500"
                  />
                  {pendingInteractions}/{3} to refresh
                </motion.span>
              )}
            </AnimatePresence>
          </div>

          <h2 className="text-2xl sm:text-3xl font-light tracking-tight text-gray-900">
            You might also like
          </h2>

          {seedProduct && (
            <p className="text-sm text-gray-400 font-light flex items-center gap-2 flex-wrap">
              Based on
              <span className="inline-flex items-center gap-1.5 text-gray-700 font-medium">
                {seedProduct.image_url && (
                  <span className="relative w-5 h-5 rounded overflow-hidden border border-gray-200 inline-block shrink-0">
                    <Image src={seedProduct.image_url} alt={seedProduct.prod_name} fill className="object-cover" sizes="20px" />
                  </span>
                )}
                {seedProduct.prod_name}
              </span>
            </p>
          )}
        </div>

        {/* Arrow buttons */}
        <div className="flex items-center gap-2 shrink-0 pt-1">
          <button
            onClick={() => scroll('left')}
            disabled={!canLeft}
            className="p-2.5 rounded-full border border-gray-200 text-gray-500 hover:text-black hover:border-gray-400 disabled:opacity-25 disabled:cursor-not-allowed transition-all"
            aria-label="Scroll left"
          >
            <ChevronLeft size={16} strokeWidth={1.5} />
          </button>
          <button
            onClick={() => scroll('right')}
            disabled={!canRight}
            className="p-2.5 rounded-full border border-gray-200 text-gray-500 hover:text-black hover:border-gray-400 disabled:opacity-25 disabled:cursor-not-allowed transition-all"
            aria-label="Scroll right"
          >
            <ChevronRight size={16} strokeWidth={1.5} />
          </button>
        </div>
      </div>

      {/* ── Scrollable track ─────────────────────────── */}
      <div className="relative">
        {/* Left gradient fade */}
        <div
          className="absolute left-0 top-0 bottom-4 w-12 bg-gradient-to-r from-white to-transparent z-10 pointer-events-none transition-opacity duration-300"
          style={{ opacity: canLeft ? 1 : 0 }}
        />
        {/* Right gradient fade */}
        <div
          className="absolute right-0 top-0 bottom-4 w-12 bg-gradient-to-l from-white to-transparent z-10 pointer-events-none transition-opacity duration-300"
          style={{ opacity: canRight ? 1 : 0 }}
        />

        <div
          ref={trackRef}
          onScroll={syncArrows}
          className="flex gap-4 overflow-x-auto pb-4"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          {loading
            ? [...Array(6)].map((_, i) => (
                <div key={i} className="flex-shrink-0 w-44 sm:w-52 animate-pulse">
                  <div className="aspect-[3/4] rounded-xl bg-gray-100 mb-3" />
                  <div className="h-2.5 bg-gray-100 rounded w-3/4 mb-2" />
                  <div className="h-2.5 bg-gray-100 rounded w-1/3" />
                </div>
              ))
            : products.slice(0, 10).map((p, i) => (
                <SliderCard key={p.article_id} product={p} index={i} onInteract={onInteract} />
              ))}
        </div>
      </div>

      {/* Footer note */}
      {!loading && products.length > 0 && (
        <p className="mt-4 text-center text-[10px] text-gray-300 tracking-widest uppercase">
          Powered by Personal MLP · Refreshes every 3 interactions
        </p>
      )}
    </motion.section>
  );
}
