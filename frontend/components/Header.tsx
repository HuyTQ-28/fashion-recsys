'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, ShoppingBag, Heart, User, ChevronDown } from 'lucide-react';

interface HeaderProps {
  userId: string;
  setUserId: (id: string) => void;
  cartCount: number;
  wishlistCount: number;
  isPersonalized: boolean;
}

export function Header({ userId, setUserId, cartCount, wishlistCount, isPersonalized }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 w-full bg-white/90 backdrop-blur-md border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Logo */}
          <div className="flex items-center gap-3">
            <motion.a
              href="/"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-lg font-bold tracking-tighter text-black leading-none"
            >
              STUDIO<span className="text-gray-300 font-light">F1</span>
            </motion.a>

            <AnimatePresence>
              {isPersonalized && (
                <motion.span
                  initial={{ opacity: 0, scale: 0.75 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.75 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 24 }}
                  className="hidden sm:inline-flex items-center gap-1 px-2.5 py-0.5 bg-emerald-50 text-emerald-700 text-[9px] font-semibold uppercase tracking-widest rounded-full border border-emerald-200"
                >
                  <Sparkles size={9} strokeWidth={2} />
                  Personalized
                </motion.span>
              )}
            </AnimatePresence>
          </div>

          {/* Nav links — desktop */}
          <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-600">
            <a href="#" className="hover:text-black transition-colors">New In</a>
            <a href="#" className="hover:text-black transition-colors">Women</a>
            <a href="#" className="hover:text-black transition-colors">Men</a>
            <a href="#" className="hover:text-black transition-colors">Sale</a>
          </nav>

          {/* Right controls */}
          <div className="flex items-center gap-1 sm:gap-3">
            {/* Profile picker */}
            <div className="relative flex items-center gap-1.5 px-3 py-1.5 rounded-full hover:bg-gray-50 transition-colors cursor-pointer group">
              <User size={15} strokeWidth={1.5} className="text-gray-500" />
              <select
                className="appearance-none bg-transparent text-xs font-medium text-gray-700 outline-none cursor-pointer pr-3"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
              >
                <option value="user_1">User 1</option>
                <option value="user_2">User 2</option>
                <option value="guest">Guest</option>
              </select>
              <ChevronDown size={11} strokeWidth={2} className="absolute right-2 text-gray-400 pointer-events-none" />
            </div>

            {/* Wishlist */}
            <button className="relative p-2 rounded-full hover:bg-gray-50 transition-colors" aria-label="Wishlist">
              <Heart size={18} strokeWidth={1.5} className="text-gray-700" />
              {wishlistCount > 0 && (
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-black text-white text-[9px] font-bold rounded-full flex items-center justify-center"
                >
                  {wishlistCount > 9 ? '9+' : wishlistCount}
                </motion.span>
              )}
            </button>

            {/* Cart */}
            <button className="relative flex items-center gap-2 px-4 py-2 bg-black text-white text-xs font-semibold tracking-wide rounded-full hover:bg-gray-900 transition-colors" aria-label="Cart">
              <ShoppingBag size={14} strokeWidth={1.5} />
              <span className="hidden sm:inline">Bag</span>
              {cartCount > 0 && (
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="ml-0.5 px-1.5 py-0.5 bg-white text-black text-[9px] font-bold rounded-full"
                >
                  {cartCount}
                </motion.span>
              )}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
