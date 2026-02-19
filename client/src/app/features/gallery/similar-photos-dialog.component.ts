import { Component, inject, signal, OnInit } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatDialogModule, MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { firstValueFrom } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { ThumbnailUrlPipe } from '../../shared/pipes/thumbnail-url.pipe';
import { FixedPipe } from '../../shared/pipes/fixed.pipe';
import { TranslatePipe } from '../../shared/pipes/translate.pipe';

interface SimilarPhoto {
  path: string;
  filename: string;
  similarity: number;
  aggregate: number | null;
  aesthetic: number | null;
  date_taken: string | null;
}

@Component({
  selector: 'app-similar-photos-dialog',
  imports: [
    MatButtonModule,
    MatDialogModule,
    MatIconModule,
    MatProgressSpinnerModule,
    ThumbnailUrlPipe,
    FixedPipe,
    TranslatePipe,
  ],
  template: `
    <h2 mat-dialog-title>{{ 'similar.title' | translate }}</h2>
    <mat-dialog-content class="!flex !flex-col gap-3 min-h-[200px]">
      @if (loading()) {
        <div class="flex items-center justify-center gap-3 py-8">
          <mat-spinner diameter="24"></mat-spinner>
          <span class="text-sm text-neutral-400">{{ 'similar.loading' | translate }}</span>
        </div>
      } @else if (!results().length) {
        <p class="text-sm text-neutral-500 text-center py-8">{{ 'similar.no_results' | translate }}</p>
      } @else {
        <div class="grid grid-cols-3 gap-2 max-h-[60vh] overflow-y-auto">
          @for (photo of results(); track photo.path) {
            <div class="relative rounded-lg overflow-hidden bg-neutral-900">
              <img [src]="photo.path | thumbnailUrl:320"
                   [alt]="photo.filename"
                   class="w-full aspect-square object-cover" />
              <div class="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1.5">
                <div class="text-xs text-white truncate">{{ photo.filename }}</div>
                <div class="flex items-center gap-2 text-[11px]">
                  <span class="text-green-400 font-medium">{{ (photo.similarity * 100) | fixed:0 }}% {{ 'similar.similarity' | translate }}</span>
                  @if (photo.aggregate != null) {
                    <span class="text-neutral-400">{{ photo.aggregate | fixed:1 }}</span>
                  }
                </div>
              </div>
            </div>
          }
        </div>
      }
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button (click)="dialogRef.close()">{{ 'dialog.cancel' | translate }}</button>
    </mat-dialog-actions>
  `,
})
export class SimilarPhotosDialogComponent implements OnInit {
  private api = inject(ApiService);
  readonly data: { photoPath: string } = inject(MAT_DIALOG_DATA);
  readonly dialogRef = inject(MatDialogRef<SimilarPhotosDialogComponent>);

  readonly loading = signal(true);
  readonly results = signal<SimilarPhoto[]>([]);

  async ngOnInit(): Promise<void> {
    try {
      const res = await firstValueFrom(
        this.api.get<{ similar: SimilarPhoto[] }>(`/similar_photos/${encodeURIComponent(this.data.photoPath)}`),
      );
      this.results.set(res.similar ?? []);
    } catch { /* ignore */ }
    this.loading.set(false);
  }
}
