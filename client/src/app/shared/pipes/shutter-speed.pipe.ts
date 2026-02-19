import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'shutterSpeed', standalone: true, pure: true })
export class ShutterSpeedPipe implements PipeTransform {
  transform(value: string | number | null | undefined): string {
    if (value == null) return '';
    const num = +value;
    if (isNaN(num) || num <= 0) return '';
    if (num >= 1) return num.toFixed(1) + 's';
    return '1/' + Math.round(1 / num);
  }
}
