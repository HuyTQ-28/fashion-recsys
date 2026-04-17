'use client';

import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Heart, ShoppingBag, Brain, TrendingUp, Zap } from 'lucide-react';
import { useEffect } from 'react';
import { Product } from '@/components/ProductCard';

interface ProductModalProps {
  product: Product | null;
  onClose: () => void;
  onFavorite: (id: string) => void;
  onCart: (id: string) => void;
  isSyncing: boolean;
}

export function ProductModal({ product, onClose, onFavorite, onCart, isSyncing }: ProductModalProps) {
  // Escape key to close
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  // Lock body scroll while open
  useEffect(() => {
    if (product) document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = ''; };
  }, [product]);

  const mockPrice = product ? (parseInt(product.article_id.slice(-4)) % 150) + 49 : 0;
  const matchPercent = product?.score !== undefined ? Math.round(product.score * 100) : null;

  const reasons = [
    'Matches your recent browsing pattern',
    'Similar to items you favorited',
    'Popular in your style category',
  ];

  return (
    <AnimatePresence>
      {product && (
        <>
          {/* Backdrop */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
          />

          {/* Panel */}
          <motion.div
            key="panel"
            initial={{ opacity: 0, y: 40, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.97 }}
            transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
            className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-0 sm:p-6 pointer-events-none"
          >
            <div className="pointer-events-auto w-full sm:max-w-3xl bg-white sm:rounded-2xl shadow-2xl overflow-hidden flex flex-col md:flex-row max-h-[95dvh]">

              {/* ── Image panel ──────────────────────── */}
              <div className="relative w-full md:w-2/5 aspect-[4/3] md:aspect-auto md:min-h-[480px] bg-gray-100 shrink-0">
                {product.image_url ? (
                  <Image
                    src={product.image_url}
                    alt={product.prod_name}
                    fill
                    priority
                    className="object-cover object-top"
                    sizes="(max-width: 768px) 100vw, 480px"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-xs text-gray-400 uppercase tracking-widest">
                    No Image
                  </div>
                )}

                {/* AI badge overlay */}
                {matchPercent !== null && (
                  <div className="absolute top-3 left-3 flex items-center gap-1.5 px-3 py-1.5 bg-white/95 backdrop-blur-sm rounded-full text-xs font-semibold text-emerald-600 shadow-md">
                    <Zap size={11} strokeWidth={2.5} />
                    {matchPercent}% Style Match
                  </div>
                )}
              </div>

              {/* ── Content panel ─────────────────────── */}
              <div className="flex flex-col flex-1 p-6 sm:p-8 overflow-y-auto">
                {/* Close button */}
                <button
                  onClick={onClose}
                  className="self-end p-2 -mt-1 -mr-1 mb-3 text-gray-400 hover:text-black hover:bg-gray-100 rounded-full transition-colors"
                  aria-label="Close"
                >
                  <X size={18} strokeWidth={1.5} />
                </button>

                {/* Meta + Name */}
                <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400 mb-2">
                  {product.product_group_name || 'Fashion'}
                  {product.colour_group_name && ` · ${product.colour_group_name}`}
                </p>
                <h2 className="text-xl sm:text-2xl font-light tracking-tight text-gray-900 leading-snug mb-2">
                  {product.prod_name}
                </h2>
                <p className="text-2xl font-semibold text-gray-900 mb-6">${mockPrice}</p>

                {/* ── Why this item ──────────────────── */}
                <div className="bg-gray-50 border border-gray-100 rounded-xl p-4 mb-6">
                  <div className="flex items-center gap-2 mb-4">
                    <Brain size={14} strokeWidth={1.5} className="text-black" />
                    <span className="text-[10px] font-semibold tracking-widest uppercase text-gray-700">
                      Why this item?
                    </span>
                  </div>

                  {matchPercent !== null ? (
                    <div className="mb-4">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs text-gray-500">Personal style match</span>
                        <span className="text-xs font-bold text-emerald-600">{matchPercent}%</span>
                      </div>
                      <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${matchPercent}%` }}
                          transition={{ duration: 0.9, ease: 'easeOut', delay: 0.15 }}
                          className="h-full bg-emerald-500 rounded-full"
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-start gap-2 mb-3">
                      <TrendingUp size={13} className="text-gray-400 mt-0.5 shrink-0" />
                      <p className="text-xs text-gray-500 leading-relaxed">
                        Matches your browsing style and profile.
                      </p>
                    </div>
                  )}

                  <ul className="flex flex-col gap-1.5">
                    {reasons.map((r, i) => (
                      <motion.li
                        key={r}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.1 + i * 0.07 }}
                        className="flex items-center gap-2 text-[11px] text-gray-500"
                      >
                        <span className="w-1 h-1 rounded-full bg-gray-300 shrink-0" />
                        {r}
                      </motion.li>
                    ))}
                  </ul>

                  {/* Syncing status */}
                  <div className="flex items-center gap-2 mt-3.5 pt-3 border-t border-gray-200">
                    <motion.span
                      animate={isSyncing
                        ? { opacity: [0.3, 1, 0.3], scale: [0.9, 1.1, 0.9] }
                        : { opacity: 1, scale: 1 }}
                      transition={{ repeat: Infinity, duration: 1.4 }}
                      className={`w-1.5 h-1.5 rounded-full shrink-0 ${isSyncing ? 'bg-amber-400' : 'bg-emerald-400'}`}
                    />
                    <span className="text-[10px] tracking-wide text-gray-400">
                      {isSyncing
                        ? 'Syncing your preferences with the AI model…'
                        : 'Style profile is up to date'}
                    </span>
                  </div>
                </div>

                {/* ── CTA Buttons ───────────────────── */}
                <div className="flex flex-col gap-3 mt-auto">
                  <motion.button
                    whileTap={{ scale: 0.97 }}
                    onClick={() => { onCart(product.article_id); onClose(); }}
                    className="w-full py-3.5 bg-black text-white text-sm font-semibold tracking-wide rounded-xl hover:bg-gray-900 active:bg-gray-800 transition-colors flex items-center justify-center gap-2"
                  >
                    <ShoppingBag size={16} strokeWidth={1.5} />
                    Add to Bag — ${mockPrice}
                  </motion.button>
                  <motion.button
                    whileTap={{ scale: 0.97 }}
                    onClick={() => { onFavorite(product.article_id); onClose(); }}
                    className="w-full py-3.5 border border-gray-200 text-sm font-medium tracking-wide rounded-xl hover:border-gray-400 active:bg-gray-50 transition-colors flex items-center justify-center gap-2 text-gray-700"
                  >
                    <Heart size={16} strokeWidth={1.5} />
                    Save to Wishlist
                  </motion.button>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
