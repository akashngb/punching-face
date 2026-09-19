"use client"

import React from "react"
import { cn } from "@/lib/utils"

const cardContents = [
  {
    title: "Beautiful Components",
    description:
      "Ruixen UI provides stunning, ready-made components built with consistent design and performance in mind.",
  },
  {
    title: "Developer Friendly",
    description:
      "Simple APIs and excellent documentation make it easy to integrate and customize Ruixen UI in your apps.",
  },
  {
    title: "Flexible Layouts",
    description:
      "Design dynamic, responsive layouts using our grid utilities and flex-based helpers. Whether you're building dashboards, landing pages, or nested components, Ruixen UI provides composable layout primitives that scale beautifully across screen sizes. With mobile-first defaults, built-in breakpoints, and utilities like col-span, row-span, gap control, and responsive spacing, your UI adapts effortlessly to every device.",
  },
  {
    title: "Dark Mode Support",
    description:
      "Every component is thoughtfully designed to work seamlessly in both light and dark themes.",
  },
  {
    title: "Fast & Lightweight",
    description:
      "Built for speed and performance, Ruixen UI ensures quick load times without sacrificing quality.",
  },
]


type PlusCardProps = React.ComponentProps<"div"> & {
  href?: string
  title?: string
  description?: string
  children?: React.ReactNode
}

const PlusCard: React.FC<PlusCardProps> = ({
  className = "",
  href,
  title,
  description,
  children,
  ...rest
}) => {
  const inner = (
    <>
      <CornerPlusIcons />
      {children ?? (
        <div className="relative z-10 space-y-2">
          <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">
            {title}
          </h3>
          <p className="text-gray-700 dark:text-gray-300">{description}</p>
        </div>
      )}
    </>
  )

  return (
    <div
      className={cn(
        "relative border border-dashed border-zinc-400 dark:border-zinc-700 rounded-lg p-6 bg-white dark:bg-zinc-950 min-h-[200px]",
        "flex flex-col justify-between",
        className
      )}
      {...rest}
    >
      {href ? (
        <a href={href} className="contents">
          {inner}
        </a>
      ) : (
        inner
      )}
    </div>
  )
}

const CornerPlusIcons = () => (
  <>
    <PlusIcon className="absolute -top-3 -left-3" />
    <PlusIcon className="absolute -top-3 -right-3" />
    <PlusIcon className="absolute -bottom-3 -left-3" />
    <PlusIcon className="absolute -bottom-3 -right-3" />
  </>
)

const PlusIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    width={24}
    height={24}
    strokeWidth="1.5"
    stroke="currentColor"
    className={`dark:text-white text-black size-6 ${className}`}
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m6-6H6" />
  </svg>
)

export { PlusCard, CornerPlusIcons, PlusIcon }

export default function RuixenBentoCards() {
  return (
    <section className="bg-white dark:bg-black dark:bg-transparent border border-gray-200 dark:border-gray-800">
      <div className="mx-auto container border border-gray-200 dark:border-gray-800 py-12 border-t-0 px-4">
        {/* Responsive Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 auto-rows-auto gap-4">
          <PlusCard
            {...cardContents[0]}
            href="https://ruixen.com/?utm_source=21stdev&utm_medium=button&utm_campaign=ruixen_bento_cards"
            className="lg:col-span-3 lg:row-span-2"
          />
          <PlusCard
            {...cardContents[1]}
            href="https://ruixen.com/?utm_source=21stdev&utm_medium=button&utm_campaign=ruixen_bento_cards"
            className="lg:col-span-2 lg:row-span-2"
          />
          <PlusCard
            {...cardContents[2]}
            href="https://ruixen.com/?utm_source=21stdev&utm_medium=button&utm_campaign=ruixen_bento_cards"
            className="lg:col-span-4 lg:row-span-1"
          />
          <PlusCard
            {...cardContents[3]}
            href="https://ruixen.com/?utm_source=21stdev&utm_medium=button&utm_campaign=ruixen_bento_cards"
            className="lg:col-span-2 lg:row-span-1"
          />
          <PlusCard
            {...cardContents[4]}
            href="https://ruixen.com/?utm_source=21stdev&utm_medium=button&utm_campaign=ruixen_bento_cards"
            className="lg:col-span-2 lg:row-span-1"
          />
        </div>

        {/* Section Footer Heading */}
        <div className="max-w-2xl ml-auto text-right px-4 mt-6 lg:-mt-20">
          <h2 className="text-4xl md:text-6xl font-bold text-black dark:text-white mb-4">
            Built for performance. Designed for flexibility.
          </h2>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Ruixen UI gives you the tools to build beautiful, high-performing websites with lightning speed. Each component is thoughtfully designed to be flexible, reusable, and accessible.
          </p>
        </div>
      </div>
    </section>
  )
}
