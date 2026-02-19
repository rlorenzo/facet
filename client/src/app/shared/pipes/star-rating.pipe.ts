import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'starArray', standalone: true, pure: true })
export class StarArrayPipe implements PipeTransform {
  private static readonly STARS = [1, 2, 3, 4, 5];
  transform(_value: unknown): number[] {
    return StarArrayPipe.STARS;
  }
}

@Pipe({ name: 'isStarFilled', standalone: true, pure: true })
export class IsStarFilledPipe implements PipeTransform {
  transform(star: number, currentRating: number | null, hoverRating: number | null | undefined): boolean {
    const effective = hoverRating ?? currentRating ?? 0;
    return star <= effective;
  }
}
