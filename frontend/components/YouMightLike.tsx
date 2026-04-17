'use client';

import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, ArrowRight } from 'lucide-react';
import { ProductCard, Product } from '@/components/ProductCard';

interface YouMightLikeProps {
  products: Product[];
  loading: boolean;
  seedProduct: Product | null;
  onInteract: (id: string, type: 'click' | 'favorite' | 'purchase') => void;
}

export function YouMightLike({ products, loading, seedProduct, onInteract }: YouMightLikeProps) {
  const show = loading || products.length > 0;
  if (!show) return null;

  return (
    <motion.section
      initial={{ opacity: 0, y: 48 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 24 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="mt-20 pt-14 border-t border-gray-100"
    >
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-10">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2.5">
            <Sparkles size={16} className="text-black" strokeWidth={1.5} />
            <span className="text-[10px] font-semibold tracking-widest uppercase text-gray-500">AI Curated</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-light tracking-tight text-gray-900">
            You might also like
          </h2>
          {seedProduct && (
            <p className="text-sm text-gray-400 font-light flex items-center gap-2">
              Based on
              <span className="flex items-center gap-1.5 text-gray-700 font-medium">
                {seedProduct.image_url && (
                  <span className="relative inline-block w-5 h-5 rounded overflow-hidden border border-gray-200">
                    <Image
                      src={seedProduct.image_url}
                      alt={seedProduct.prod_name}
                      fill
                      className="object-cover"
                      sizes="20px"
                    />
                  </span>
                )}
                {seedProduct.prod_name}
              </span>
            </p>
          )}
        </div>

        {!loading && products.length > 0 && (
          <button className="flex items-center gap-1.5 text-sm font-medium text-gray-500 hover:text-black transition-colors self-start sm:self-auto">
            See all
            <ArrowRight size={15} strokeWidth={1.5} />
          </button>
        )}
      </div>

      {/* Skeleton or Grid */}
      {loading ? (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-10">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="animate-pulse flex flex-col gap-4">
              <div className="bg-gray-100 aspect-[3/4] rounded-md" />
              <div className="space-y-2.5">
                <div className="h-2.5 bg-gray-100 rounded w-3/4" />
                <div className="h-2.5 bg-gray-100 rounded w-1/3" />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <motion.div
          layout
          className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-12"
        >
          <AnimatePresence mode="popLayout">
            {products.slice(0, 8).map((p, i) => (
              <motion.div
                key={p.article_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.06 }}
              >
                <ProductCard
                  product={p}
                  onClick={(id) => onInteract(id, 'click')}
                  onFavorite={(id) => onInteract(id, 'favorite')}
                  onCart={(id) => onInteract(id, 'purchase')}
                />
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>
      )}

      {/* AI explanation footer */}
      {!loading && products.length > 0 && (
        <p className="mt-10 text-center text-xs text-gray-300 tracking-wide">
          Recommendations update in real-time as you browse · Powered by Personal MLP
        </p>
      )}
    </motion.section>
  );
}
