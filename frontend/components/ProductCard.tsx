'use client';

import Image from 'next/image';
import { Heart, ShoppingBag } from 'lucide-react';
import { motion } from 'framer-motion';

export interface Product {
  article_id: string;
  prod_name: string;
  image_url: string;
  score?: number;
  product_group_name?: string;
  colour_group_name?: string;
}

interface Props {
  product: Product;
  onClick: (id: string) => void;
  onFavorite: (id: string) => void;
  onCart: (id: string) => void;
  compact?: boolean;
}

export function ProductCard({ product, onClick, onFavorite, onCart, compact = false }: Props) {
  // Generate a mock price based on article_id for UI aesthetics
  const mockPrice = (parseInt(product.article_id.slice(-4)) % 150) + 49;

  return (
    <motion.div 
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={`group relative flex flex-col ${compact ? 'gap-2' : 'gap-4'}`}
    >
      <div 
        className="relative w-full aspect-[3/4] cursor-pointer overflow-hidden bg-gray-100/50 rounded-md"
        onClick={() => onClick(product.article_id)}
      >
        {product.image_url ? (
          <Image
            src={product.image_url}
            alt={product.prod_name || 'Product'}
            fill
            priority
            className="object-cover object-top transition-all duration-500 ease-out group-hover:scale-105"
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 25vw"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-xs tracking-widest text-gray-400 uppercase">
            No Image
          </div>
        )}
        
        {/* Overlay Actions */}
        <div className="absolute bottom-0 left-0 right-0 p-4 translate-y-full opacity-0 transition-all duration-300 ease-out group-hover:translate-y-0 group-hover:opacity-100 bg-gradient-to-t from-black/20 to-transparent flex justify-between items-end">
          <motion.button 
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={(e) => { 
              e.stopPropagation(); 
              onFavorite(product.article_id); 
            }}
            className="p-3 bg-white/95 backdrop-blur text-black rounded-full hover:bg-white hover:text-red-500 transition-colors shadow-lg"
            aria-label="Add to Favorites"
          >
            <Heart size={18} strokeWidth={1.5} />
          </motion.button>
          
          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={(e) => { 
              e.stopPropagation(); 
              onCart(product.article_id); 
            }}
            className="flex items-center gap-2 px-5 py-3 bg-black/95 backdrop-blur text-white text-[11px] font-semibold tracking-widest uppercase rounded-full hover:bg-black transition-colors shadow-lg"
          >
            <ShoppingBag size={16} strokeWidth={1.5} />
            <span>Add to bag</span>
          </motion.button>
        </div>
      </div>
      
      {/* Product Info */}
      <div className={`flex flex-col px-1 ${compact ? 'gap-1' : 'gap-1.5'}`}>
        <div className="flex justify-between items-start gap-4">
          <button
            type="button"
            onClick={() => onClick(product.article_id)}
            className="text-left"
            title={product.prod_name}
          >
            <h3 className="font-medium text-sm text-gray-900 tracking-tight leading-snug line-clamp-1 hover:text-black transition-colors">
              {product.prod_name || 'Unknown Product'}
            </h3>
          </button>
          <span className="text-sm font-medium text-gray-900 tracking-tight">${mockPrice}</span>
        </div>
        
        <div className="flex items-center gap-2 text-xs text-gray-500 tracking-wide uppercase">
          <span>{product.product_group_name || 'Fashion'}</span>
          {product.score !== undefined && (
            <>
              <span className="w-1 h-1 rounded-full bg-gray-300"></span>
              <span className="text-emerald-600 font-medium lowercase flex items-center gap-1">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
                {Math.round(product.score * 100)}% Match
              </span>
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
}