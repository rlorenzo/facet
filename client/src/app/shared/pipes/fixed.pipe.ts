import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'fixed', standalone: true, pure: true })
export class FixedPipe implements PipeTransform {
  transform(value: number | null | undefined, digits: number = 1): string {
    if (value == null) return '';
    return value.toFixed(digits);
  }
}
