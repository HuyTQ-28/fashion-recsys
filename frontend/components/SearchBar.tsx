'use client';

import { useState, useRef, useEffect } from 'react';
import { Search, X, Loader2 } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  loading?: boolean;
}

export function SearchBar({ onSearch, loading }: SearchBarProps) {
  const [value, setValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value;
    setValue(v);
    // Debounce 500ms so we don't fire on every keystroke
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      onSearch(v.trim());
    }, 500);
  };

  const handleClear = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setValue('');
    onSearch('');
    inputRef.current?.focus();
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (debounceRef.current) clearTimeout(debounceRef.current);
    onSearch(value.trim());
  };

  useEffect(() => () => { if (debounceRef.current) clearTimeout(debounceRef.current); }, []);

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto" role="search">
      <div className="relative flex items-center group">
        <div className="absolute left-5 text-gray-400 pointer-events-none transition-colors group-focus-within:text-black">
          {loading
            ? <Loader2 size={18} strokeWidth={1.5} className="animate-spin" />
            : <Search size={18} strokeWidth={1.5} />}
        </div>
        <input
          ref={inputRef}
          type="search"
          value={value}
          onChange={handleChange}
          placeholder="Search for products, brands, styles..."
          className="w-full pl-13 pr-12 py-4 bg-gray-50 border border-gray-200 rounded-2xl text-sm text-black placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-black/10 focus:border-gray-400 transition-all"
        />
        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-5 p-1 text-gray-400 hover:text-black transition-colors rounded-full hover:bg-gray-100"
            aria-label="Clear search"
          >
            <X size={15} strokeWidth={2} />
          </button>
        )}
      </div>
    </form>
  );
}
