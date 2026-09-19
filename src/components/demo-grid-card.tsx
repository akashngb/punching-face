import { GridCard } from "@/components/ui/grid-card";
import { Grid2x2Check } from 'lucide-react';

export default function DemoGridCard() {
     return (
        <div className="relative flex min-h-screen w-full items-center justify-center">
            <GridCard className="size-72">
                <Grid2x2Check className="text-foreground/80 relative size-8" />
                <div className="relative">
                    <span className="text-foreground/80 text-lg font-medium">
                        Grid Card
                    </span>
                    <p className="text-muted-foreground text-sm">
                        Have you ever wondered how to create a grid card with a hover effect?
                    </p>
                </div>
            </GridCard>
        </div>
    );
}
